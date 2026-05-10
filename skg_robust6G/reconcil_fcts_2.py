import numpy as np
from quantize import quantizer_linear

import multiprocessing

from polar_gaussian import reliability_fct
from information_reconciliation import InformationReconciliation

def recon_id(quant_up, quant_down, code, N_code, rate, snr_dB, N_samples, rate_id): 

    """
    Parameters: 
        H_up: uplink array of size (Number of Nsamples, N_code)
        H_down: downlink array of size Number of (Nsamples, N_code)
        code (str): Error correction code to use (e.g., 'Polar_CRC').
        rate (float): Code rate
        n_bits (int): Number of bits for quantization.
        N_code (int): Codelength 
        processes (int, optional): Number of parallel processes to use. If None, defaults to the number of CPU cores.
    Returns:
        error_prob (float): Error probability after reconciliation      
    """
    code_rate = rate[rate_id]
    # print("rate[",rate_id,"] = ", code_rate)

    eta = np.zeros((N_samples, ))

    recon_up = np.zeros((N_samples, int(N_code*code_rate)-1))
    recon_dw = np.zeros((N_samples, int(N_code*code_rate)-1))
 
    for s in range(N_samples):
        # print("Sample ", s)
        # if (s + 1) % 500 == 0 or (s + 1) == 1:
        #     print("Sample ", s+1)

        q0 = quant_up[s,:]
        q1 = quant_down[s,:]

        #Reconciliation 
        ir_ab = InformationReconciliation(code, code_rate, code_one=q0, code_two=q1, SNR_dB = snr_dB)
        r1, r2 = ir_ab.implement_SW()
        r1 = np.array(r1)
        r2 = np.array(r2)

        recon_up[s,:] = r1
        recon_dw[s,:] = r2

        eta_s = (r1 != r2)   
        eta[s]  =  eta_s.sum()  
    
    error_prob = np.sum(eta)/(N_samples*len(r1))

    return error_prob, recon_up, recon_dw


def parallel_fct(H_up, H_down, code, rate, snr_dB, n_bits, N_code, processes=None):

    """
    Parameters: 
        H_up (np.ndarray): Uplink array of size (N_samples, N_code).
        H_down (np.ndarray): Downlink array of size (N_samples, N_code).
        code (str): Error correction code to use (e.g., 'Polar_CRC').
        rate (float or list): Code rate(s).
        n_bits (int): Number of bits for quantization.
        N (int): Codelength for n_bits = 1 (Codelength = n_bits * N).
        processes (int, optional): Number of parallel processes to use. If None, defaults to the number of CPU cores.
    Returns:
        bmr (np.ndarray): Bit mismatch rates for each rate.
        error_prob (np.ndarray): Error probabilities for each rate.
    """
    # Generate the bit channels of the polar code (Polar code construction)
    Q_snr= {}
    Q_snr[int(snr_dB)] = reliability_fct(snr_dB, N_code, rate = rate[0])
    np.savez("bit_channel.npz", Q_snr = Q_snr, allow_pickle=True)

    quant_up = quantizer_linear(H_up, n_bits, return_bits=True, gray=True)[0]
    quant_down = quantizer_linear(H_down, n_bits, return_bits=True, gray=True)[0]

    N_samples = 1
    #Reshape the data
    quant_up = quant_up[:N_code].reshape(1, N_code)
    quant_down = quant_down[:N_code].reshape(1, N_code)

    # Argument list for each rate
    args = [
        (quant_up, quant_down, code, N_code, rate, snr_dB, N_samples, rate_id)
        for rate_id in range(len(rate))
    ]

    pool_kwargs = {}
    if processes is not None:
        pool_kwargs['processes'] = processes

    with multiprocessing.Pool(**pool_kwargs) as pool:
        results = pool.starmap(recon_id, args)

    #Extract results
    zip_results = list(zip(*results))
    error_prob = np.array(zip_results[0])  #error probabilities for each rate
    recon_up = list(zip_results[1])  #reconciled bits for Alice for each rate
    recon_dw = list(zip_results[2])  #reconciled bits for Bob for each rate

    return error_prob, recon_up, recon_dw  #, N_samples



