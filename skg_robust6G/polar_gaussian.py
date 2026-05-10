import numpy as np

def reliability_fct(SNR_dB, N, rate = None):
    """
    Polar code construction using density evolution with the Gaussian Approximation. Each channel can have different parameters.

    Parameters
    ----------
    z0: ndarray<float>, float
        a vector of the initial mean likelihood densities, 4 * E_b/N_o.
        > Note that this SNR should be normalised using `get_normalised_SNR` in `PolarCode`

    Returns
    ----------
    ndarray<int>, ndarray<int>
        channel reliabilities in log-domain (least reliable first), and the frozen indices

    -------------
    **References:**

    * Trifonov, P. (2012). Efficient Design and Decoding of Polar Codes. IEEE Transactions on Communications, 60(11), 3221–3227. https://doi.org/10.1109/TCOMM.2012.081512.110872

    * Vangala, H., Viterbo, E., & Hong, Y. (2015). A Comparative Study of Polar Code Constructions for the AWGN Channel. arXiv.org. Retrieved from http://search.proquest.com/docview/2081709282/

    """
    n = int(np.log2(N))  # number of bits per index

    z = np.zeros((N, n + 1))

    EbN0 = 10 ** (SNR_dB / 10)  # convert SNR from dB to linear scale

    #z0  # initial channel states
    if rate is None:
        z[:, 0] =  EbN0
    else:
        z[:, 0] = rate*EbN0 # normalised message signal energy by the rate

    for j in range(1, n + 1):
        u = 2 ** j  # number of branches at depth j
        # loop over top branches at this stage
        for t in range(0, N, u):
            for s in range(int(u / 2)):
                k = t + s
                z_top = z[k, j - 1]
                z_bottom = z[k + int(u / 2), j - 1]

                z[k, j] = phi_inv(1 - (1 - phi(z_top)) * (1 - phi(z_bottom)))
                z[k + int(u / 2), j] = z_top + z_bottom

    m = np.array([logQ_Borjesson(0.707*np.sqrt(z[i, n])) for i in range(N)])
    reliabilities = np.argsort(-m, kind='mergesort')    # ordered by least reliable to most reliable
    # frozen = np.argsort(m, kind='mergesort')[K:]     # select N-K least reliable channels
    #FERest = self.FER_estimate(frozen, m)
    # z = m
    
    # Q = reliabilities
    # np.savez("bit_channel.npz", Q = Q, allow_pickle=True)

    return reliabilities


def phi_residual(x, val):
    return phi(x) - val

def phi(x):
    if x < 10:
        y = -0.4527 * (x ** 0.86) + 0.0218
        y = np.exp(y)
    else:
        y = np.sqrt(3.14159 / x) * (1 - 10 / (7 * x)) * np.exp(-x / 4)
    return y

def phi_inv(y):
    return bisection(y, 0, 10000)

def bisection(val, a, b):
    c = a
    while (b - a) >= 0.01:
        # check if middle point is root
        c = (a + b) / 2
        if (phi_residual(c, val) == 0.0):
            break

        # choose which side to repeat the steps
        if (phi_residual(c, val) * phi_residual(a, val) < 0):
            b = c
        else:
            a = c
    return c

def logQ_Borjesson(x):
    a = 0.339
    b = 5.510
    half_log2pi = 0.5 * np.log(2 * np.pi)
    if x < 0:
        x = -x
        y = -np.log((1 - a) * x + a * np.sqrt(b + x * x)) - (x * x / 2) - half_log2pi
        y = np.log(1 - np.exp(y))
    else:
        y = -np.log((1 - a) * x + a * np.sqrt(b + x * x)) - (x * x / 2) - half_log2pi
    return y