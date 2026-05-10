"""
Secret-Key Generation (SKG) wrapper.

Drives the reconciliation pipeline from skg_robust6G with positions taken
from the demonstrator's placed nodes (User -> Alice, Eavesdropper -> Eve)
instead of a hardcoded `idx`. Alice's reconciled bitstream is hashed via
Davies-Meyer / AES-128 to produce a 128-bit secret key. Eve's matching
percentage is computed by quantising her own CSI with the same quantizer
and comparing to Alice's reconciled bits before hashing.

Outputs:
    alice_key_bits    : (128,) int array of 0/1
    alice_key_hex     : hex string for display
    eve_key_bits      : (128,) int array of 0/1, or None if no Eve placed
    eve_match_pct     : float in [0, 100] - bit-match between Alice and Eve
                        before hashing (the spatial decorrelation metric)
    eve_key_match_pct : same after hashing - bit match of the 128-bit keys
    reconciliation_pct: float in [0, 100] - 100 * (1 - error_prob)

The reconciliation step takes ~25-40 s per call so this should be run from
a worker thread; the GUI polls the result via a queue.
"""

from __future__ import annotations

import os
import sys
import numpy as np
from numpy.random import default_rng
from Crypto.Cipher import AES


# ----------------------------------------------------------------------------
# Davies-Meyer / AES-128 (Python port of AES_matlab/davies_meyer_hash.m)
# ----------------------------------------------------------------------------
def _bits_to_bytes(bits_128: np.ndarray) -> bytes:
    """128 bits (0/1, MSB first per byte) -> 16 bytes. Matches MATLAB."""
    arr = np.asarray(bits_128).astype(np.uint8).reshape(16, 8)
    return bytes(np.packbits(arr, axis=1).flatten())


def _bytes_to_bits(b: bytes) -> np.ndarray:
    return np.unpackbits(np.frombuffer(b, dtype=np.uint8)).astype(int)


def davies_meyer_hash(bitstream: np.ndarray) -> tuple[bytes, np.ndarray]:
    """Iterated Davies-Meyer compression with AES-128.

    H_0 = first 128-bit block (as 16 bytes)
    H_i = AES_encrypt(key=m_i, plaintext=H_{i-1}) XOR H_{i-1}

    Returns (key_bytes (16,), key_bits (128,) MSB-first per byte).
    """
    bits = np.asarray(bitstream).astype(int).flatten()
    if bits.size == 0 or bits.size % 128 != 0:
        raise ValueError(
            f"davies_meyer_hash: bitstream length {bits.size} not multiple of 128"
        )
    n_blocks = bits.size // 128
    if n_blocks < 2:
        raise ValueError(
            "davies_meyer_hash: need >= 2 blocks (256 bits): one H_0 + one message"
        )
    blocks = bits.reshape(n_blocks, 128)
    H = _bits_to_bytes(blocks[0])
    for i in range(1, n_blocks):
        m = _bits_to_bytes(blocks[i])
        cipher = AES.new(m, AES.MODE_ECB)
        E_out = cipher.encrypt(H)
        H = bytes(a ^ b for a, b in zip(E_out, H))
    return H, _bytes_to_bits(H)


# ----------------------------------------------------------------------------
# SKGEngine
# ----------------------------------------------------------------------------
class SKGEngine:
    """Secret-Key Generation engine.

    The heavy lifting (channel observation -> linear quantization ->
    Polar-CRC reconciliation -> AES Davies-Meyer hash) is delegated to the
    `skg_robust6G` package. We only need to point it at the right paths and
    arrange the inputs.

    Constructor parameters mirror the reference `main_recon.py` defaults so
    the demonstrator behaves identically to the standalone script.
    """

    def __init__(self, skg_pkg_dir: str, dataset_path: str | None = None):
        self.skg_pkg_dir = skg_pkg_dir
        self.dataset_path = dataset_path or os.path.join(
            skg_pkg_dir, 'dataset', 'data_ULA_skg.npz'
        )
        self._fns_loaded = False
        self._channel_obs = None
        self._parallel_fct = None
        self._quantizer_linear = None
        self._csi_up = None
        self._csi_dw = None
        self._positions_up = None
        self._positions_dw = None

    # ------------------------------------------------------------------
    # Lazy import: the skg package needs to be on sys.path at first use.
    # Note: the demonstrator has its own top-level `utils` package, so a
    # plain `from utils import channel_obs` finds that one first and the
    # SKG's `utils.py` is shadowed. We load each SKG module directly by
    # absolute file path with importlib to bypass the name collision.
    # ------------------------------------------------------------------
    def _ensure_loaded(self):
        if self._fns_loaded:
            return
        import importlib.util

        if self.skg_pkg_dir not in sys.path:
            sys.path.insert(0, self.skg_pkg_dir)

        def _load(mod_name: str, file_name: str):
            path = os.path.join(self.skg_pkg_dir, file_name)
            if not os.path.isfile(path):
                raise FileNotFoundError(
                    f"Could not locate SKG module at {path!r}. "
                    f"Is the skg_robust6G package complete?"
                )
            # Use a unique fully-qualified name to avoid clashing with the
            # demonstrator's own modules of the same short name (utils etc.).
            full_name = f"_skg_pkg.{mod_name}"
            spec = importlib.util.spec_from_file_location(full_name, path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[full_name] = module
            # The SKG modules import each other by their short names
            # ("from quantize import ...", "from polar_gaussian import ..."),
            # so we ALSO register them under their short name in sys.modules
            # before exec_module so cross-imports succeed.
            sys.modules[mod_name] = module
            spec.loader.exec_module(module)
            return module

        # Order matters: load lower-level deps before things that import them.
        _load('GF_fields_dict_reverse', 'GF_fields_dict_reverse.py')
        _load('bch', 'bch.py')
        _load('polar_gaussian', 'polar_gaussian.py')
        _load('polar', 'polar.py')
        _load('information_reconciliation', 'information_reconciliation.py')
        _load('quantize', 'quantize.py')
        skg_utils = _load('skg_utils', 'utils.py')   # the SKG one
        # Override sys.modules['utils'] - which the SKG module set when it
        # was loaded - so the demonstrator's own `utils` package isn't
        # permanently shadowed. We've already extracted what we need from
        # skg_utils, so removing the alias is safe.
        if sys.modules.get('utils') is skg_utils:
            del sys.modules['utils']
        reconcil_fcts = _load('reconcil_fcts_2', 'reconcil_fcts_2.py')

        self._channel_obs = skg_utils.channel_obs
        # Note: we deliberately do NOT use parallel_fct - its multiprocessing
        # Pool can't pickle our dynamically-loaded modules. We replicate its
        # logic in-process with `_run_reconciliation` below using the same
        # functions it would call.
        self._recon_id = reconcil_fcts.recon_id
        self._reliability_fct = sys.modules['polar_gaussian'].reliability_fct
        self._quantizer_linear = sys.modules['quantize'].quantizer_linear

        data = np.load(self.dataset_path)
        self._csi_up = data['csi_up']
        self._csi_dw = data['csi_dw']
        self._positions_up = data['UE_positions_up']
        self._positions_dw = data['UE_positions_dw']
        self._fns_loaded = True

    def _run_reconciliation(self, csi_up_obs_alice, csi_dw_obs_alice,
                             code, rate, snr_db, n_bits, n_code):
        """In-process equivalent of skg's parallel_fct - no multiprocessing."""
        # 1. Build the polar bit-channel and persist it; the inner
        #    InformationReconciliation reads it from cwd.
        q_snr = {int(snr_db): self._reliability_fct(snr_db, n_code, rate=rate[0])}
        np.savez('bit_channel.npz', Q_snr=q_snr, allow_pickle=True)

        # 2. Quantise both Alice's uplink and downlink observations.
        quant_up = self._quantizer_linear(csi_up_obs_alice, n_bits,
                                           return_bits=True, gray=True)[0]
        quant_dw = self._quantizer_linear(csi_dw_obs_alice, n_bits,
                                           return_bits=True, gray=True)[0]
        # parallel_fct truncates to N_code samples and reshapes to (1, N_code).
        quant_up = np.asarray(quant_up[:n_code]).reshape(1, n_code)
        quant_dw = np.asarray(quant_dw[:n_code]).reshape(1, n_code)

        # 3. Run reconciliation for each rate (we only support one for now).
        n_samples = 1
        results = []
        for rate_id in range(len(rate)):
            results.append(self._recon_id(
                quant_up, quant_dw, code, n_code, rate, snr_db,
                n_samples, rate_id))
        # Same shape as parallel_fct return: (error_prob, recon_up_list, recon_dw_list)
        zip_results = list(zip(*results))
        error_prob = np.array(zip_results[0])
        recon_up = list(zip_results[1])
        recon_dw = list(zip_results[2])
        return error_prob, recon_up, recon_dw

    # ------------------------------------------------------------------
    # Position -> grid index
    # ------------------------------------------------------------------
    def _snap_idx(self, xy: tuple[float, float], positions: np.ndarray) -> int:
        d = np.sqrt((positions[:, 0] - xy[0]) ** 2
                    + (positions[:, 1] - xy[1]) ** 2)
        return int(np.argmin(d))

    # ------------------------------------------------------------------
    # Run one full SKG cycle for a given (alice_xy, eve_xy).
    # ------------------------------------------------------------------
    def run(self, alice_xy: tuple[float, float],
            eve_xy: tuple[float, float] | None = None,
            snr_db: float = 30.0,
            n_code: int = 16384,
            n_bits: int = 2,
            rate=None,
            code: str = 'Polar_CRC',
            seed: int = 42,
            progress_cb=None) -> dict:
        """Run reconciliation + key derivation. Safe to call from a worker.

        Returns a result dict with the fields described in this module's
        docstring.
        """
        if rate is None:
            rate = np.array([0.1])

        self._ensure_loaded()

        if progress_cb:
            progress_cb('Snapping positions to dataset grid...')

        alice_idx = self._snap_idx(alice_xy, self._positions_up)
        if eve_xy is not None:
            eve_idx = self._snap_idx(eve_xy, self._positions_up)
        else:
            eve_idx = None

        # ------------------------------------------------------------------
        # Generate Alice's reconciled bits via the skg package.
        # The skg package's bit_channel.npz cache lives in cwd, so
        # temporarily chdir into the skg dir.
        # ------------------------------------------------------------------
        rng = default_rng(seed=seed)
        prev_cwd = os.getcwd()
        try:
            os.chdir(self.skg_pkg_dir)

            if progress_cb:
                progress_cb('Adding receiver noise (channel observation)...')
            csi_up_obs = self._channel_obs(self._csi_up, snr_db, rng)
            csi_dw_obs = self._channel_obs(self._csi_dw, snr_db, rng)

            if progress_cb:
                progress_cb('Running Polar-CRC reconciliation (~25 s)...')
            error, recon_up, recon_dw = self._run_reconciliation(
                csi_up_obs[:, alice_idx], csi_dw_obs[:, alice_idx],
                code, rate, snr_db, n_bits, n_code)
        finally:
            os.chdir(prev_cwd)

        alice_bits = np.asarray(recon_up[0]).astype(int).flatten()
        bob_bits = np.asarray(recon_dw[0]).astype(int).flatten()

        # Truncate to a multiple of 128 bits (per MATLAB main_skg.m).
        n_full = (alice_bits.size // 128) * 128
        if n_full < 256:
            raise RuntimeError(
                f"Reconciled bitstream too short ({alice_bits.size} bits); "
                "need >= 256 for Davies-Meyer."
            )
        alice_bits = alice_bits[:n_full]
        bob_bits = bob_bits[:n_full]

        # Hash Alice's bits to the 128-bit key.
        if progress_cb:
            progress_cb('Hashing reconciled bits with Davies-Meyer / AES-128...')
        alice_key_bytes, alice_key_bits = davies_meyer_hash(alice_bits)
        bob_key_bytes, bob_key_bits = davies_meyer_hash(bob_bits)

        reconciliation_pct = 100.0 * (1.0 - float(np.mean(error)))

        # ------------------------------------------------------------------
        # Eve's branch: she does NOT participate in reciprocity, so she
        # quantises her own observed CSI with the same quantizer (n_bits=2,
        # gray=True) and compares to Alice's reconciled bits.
        # ------------------------------------------------------------------
        eve_pre_recon_match_pct = None
        eve_key_bits = None
        eve_key_bytes = None
        eve_key_match_pct = None
        if eve_idx is not None:
            if progress_cb:
                progress_cb('Computing Eve match percentage...')
            try:
                os.chdir(self.skg_pkg_dir)
                eve_csi_up = self._channel_obs(self._csi_up, snr_db, default_rng(seed + 1))
                eve_q = self._quantizer_linear(eve_csi_up[:, eve_idx],
                                                n_bits, return_bits=True,
                                                gray=True)[0]
            finally:
                os.chdir(prev_cwd)

            eve_q = np.asarray(eve_q).astype(int).flatten()[:n_full]
            if eve_q.size < n_full:
                eve_q_padded = np.zeros(n_full, dtype=int)
                eve_q_padded[:eve_q.size] = eve_q
                eve_q = eve_q_padded
            eve_pre_recon_match_pct = 100.0 * float(np.mean(eve_q == alice_bits))

            # Hash Eve's quantised bits to attempt a key.
            try:
                eve_key_bytes, eve_key_bits = davies_meyer_hash(eve_q)
            except Exception:
                eve_key_bits = None
            if eve_key_bits is not None:
                eve_key_match_pct = 100.0 * float(
                    np.mean(eve_key_bits == alice_key_bits)
                )

        return dict(
            alice_idx=alice_idx,
            eve_idx=eve_idx,
            alice_key_bits=alice_key_bits,
            alice_key_bytes=alice_key_bytes,
            alice_key_hex=alice_key_bytes.hex(),
            eve_key_bits=eve_key_bits,
            eve_key_bytes=eve_key_bytes,
            eve_key_hex=eve_key_bytes.hex() if eve_key_bytes else None,
            eve_pre_recon_match_pct=eve_pre_recon_match_pct,
            eve_key_match_pct=eve_key_match_pct,
            reconciliation_pct=reconciliation_pct,
            n_recon_bits=int(n_full),
            error_prob=float(np.mean(error)),
            snr_db=float(snr_db),
            alice_bob_match=bool(np.array_equal(alice_key_bits, bob_key_bits)),
        )
