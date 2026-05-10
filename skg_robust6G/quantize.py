import numpy as np

def gray_code(n):
    return n ^ (n >> 1)

def indices_to_bits(indices, n_bits):
    # Vectorized integer: bit array 
    indices = np.asarray(indices, dtype=np.uint64)
    # out = np.zeros(indices.size * n_bits, dtype=np.uint8)
    out = np.zeros(indices.size * n_bits, dtype=np.int32)
    for k in range(n_bits):
        # (n_bits-1-k)th bit
        out[k::n_bits] = (indices >> (n_bits - 1 - k)) & 1
    return out

def quantizer_linear(input_data, n_bits=None, return_bits=False, gray=True):

    if n_bits is not None:
        n_bits = int(n_bits)

    quantized, mse, boundaries, levels, interval_indices = quantizer_fct(input_data, n_bits)

    if return_bits:
        if n_bits is None:
            raise ValueError("n_bits required when return_bits=True.")
        codes = interval_indices
        if gray:
            # vectorized gray code
            codes = gray_code(codes)
        bits = indices_to_bits(codes, n_bits)
        return  bits, interval_indices
    else:
        return quantized, mse


def quantizer_fct(input_data, n_bits):

    input_data = input_data.flatten()

    x_min = input_data.min()
    x_max = input_data.max()
    
    L = 2**n_bits  #number of levels
    delta = (x_max - x_min) / L  #step size
    # print("delta ", delta)
    
    # Decision boundaries
    boundaries = x_min + np.arange(L + 1) * delta
    
    # Reconstruction levels
    levels = (boundaries[:L] + boundaries[1:]) / 2 
    # levels = x_min + (np.arange(L) + 1/2) * delta
    
    # Find which interval each value belongs to
    interval_indices = np.digitize(input_data, boundaries[:-1], right=False) - 1
    # print("interval_indices ", interval_indices)
    
    # # Clip to valid range [0, L-1]
    # interval_indices = np.clip(interval_indices, 0, L - 1)
    # print("interval_indices ", interval_indices)
    
    # Mapping to reconstruction levels
    quantized = levels[interval_indices]

    #Mean Squared Error 
    mse = np.mean((input_data - quantized) ** 2)
    # print("MSE ", mse)

    return quantized, mse, boundaries, levels, interval_indices


def quant_test(input_data, nbr_bits):
    
    mse = np.zeros((len(nbr_bits),))
    for ii in range(len(nbr_bits)):
        # print("n_bits ", ii)
        mse[ii] = quantizer_fct(input_data, nbr_bits[ii])[1]

    return mse
        

#bit mistmatch rate
def bit_mistmatch_rate(bits_A, bits_B):

    if bits_A.shape != bits_B.shape:
        raise ValueError("bits_A and bits_B must have the same shape.")
    
    bits_A = bits_A.flatten()
    bits_B = bits_B.flatten()

    bmr = (bits_A != bits_B).sum() / len(bits_A)
    
    return bmr



