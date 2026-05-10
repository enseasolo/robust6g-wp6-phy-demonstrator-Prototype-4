import numpy as np
from numpy.random import default_rng
from utils import reconcil

# Choose an index for the UE position (Alice) in the grid (between 0 and  575)
idx = 200

error, alice_bits, bob_bits = reconcil(idx, 
                                  snr_dB = 30,
                                  N_code = 16384,
                                  n_bits = 2,
                                  rate = np.array([0.1]),  
                                  code = 'Polar_CRC',
                                  rng = default_rng(seed = 42)
                                 )

print(f"alice_bits: {alice_bits.shape}")
print(f"bob_bits: {bob_bits.shape}")