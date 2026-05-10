import numpy as np

from numpy.random import default_rng
from reconcil_fcts_2 import parallel_fct
from scipy.io import savemat

def channel_obs(H, SNRdB, rng): 

    CSI = np.zeros((2*H.shape[0]*H.shape[1], H.shape[2]))

    N_ant = H.shape[0]
    N_subb  = H.shape[1]
    
    for ii in range(H.shape[2]):

        H_ii = H[:,:,ii]

        if SNRdB is not None:
            gain_ = np.trace(np.dot(np.conj(H_ii).T,H_ii)) #channel gain

            #variance of the receiver noise
            sigma_Z = np.sqrt( np.real(gain_) / (N_ant*N_subb*10**(SNRdB/10)) ) 

            Z = sigma_Z*(rng.standard_normal(size=(N_ant, N_subb)) + 
                        1j * rng.standard_normal(size=(N_ant, N_subb))) / np.sqrt(2)
            #channel observation
            H_z = H_ii + Z  
        else:
            H_z = H_ii
            
        H_z = H_z.flatten()  
        H_z = np.concatenate((np.real(H_z), np.imag(H_z)))
        H_z = H_z - np.mean(H_z) #center the data around zero

        CSI[:,ii] = H_z 
    
    return CSI

def reconcil(idx, 
        snr_dB = 30,
        N_code = 16384,
        n_bits = 2,
        rate = np.array([0.1]),  
        code = 'Polar_CRC',
        rng = default_rng(seed = 42)
        ):

    #load CSI data for uplink and downlink channels (Alice and Bob)
    csi_UE_grid = np.load("dataset/data_ULA_skg.npz")
    csi_up = csi_UE_grid['csi_up'] #uplink
    csi_dw = csi_UE_grid['csi_dw'] #downlink

    # UE_positions_up = csi_UE_grid['UE_positions_up']
    # UE_positions_dw = csi_UE_grid['UE_positions_dw']
    # print(f"\nPosition of the UE, {UE_positions_up[idx, :]} meter") #position of the UE (Alice)

    csi_up = channel_obs(csi_up, snr_dB, rng)
    csi_dw = channel_obs(csi_dw, snr_dB, rng)

    error, recon_up, recon_dw = parallel_fct(csi_up[:, idx], csi_dw[:, idx], code, 
                                             rate, snr_dB, n_bits, N_code, processes=None)
    
    alice_bits = recon_up[0]
    bob_bits = recon_dw[0]
    print(f"\nError rate: {error}")
    print(f"\nTest equality (after reconciliation): {np.array_equal(alice_bits, bob_bits)}")

    N_128 = int(alice_bits.shape[1]/128)
    alice_bits = alice_bits[:,:N_128*128]
    bob_bits = bob_bits[:,:N_128*128]

    #save the reconciled bits for Alice and Bob (savemat)
    savemat(f"reconciled_bits/reconciled_bits.mat", {"idx": idx,"alice_bits": alice_bits, "bob_bits": bob_bits})

    return error, alice_bits, bob_bits