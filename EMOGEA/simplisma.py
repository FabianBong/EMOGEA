import numpy as np
import pandas as pd


def simplisma(
    expression_matrix: pd.DataFrame,
    number_of_components: int = 15,
    noise_factor: float = 0.009,
) -> dict:
    """
    Simple-to-use interactive self-modeling mixture analysis (SIMPLISMA) method implementation.

    Parameters
    ----------
    expression_matrix : pd.DataFrame
        A numpy array with genes as rows and samples as columns.

    number_of_components : int
        Number of components to project onto.

    noise_factor : float
        Noise factor.

    Returns
    -------
    dict
    """

    # assign to variables
    data = expression_matrix.T.to_numpy()
    comp = number_of_components

    # initialize algoirthm
    data[np.isnan(data)] = 0
    r, c = data.shape
    alpha = noise_factor
    ipure = np.zeros(comp)
    var_sim = np.zeros(c)
    coo = np.zeros((comp, comp))
    lambda_i = np.sqrt(np.sum(data, axis=0) ** 2 / data.shape[0])

    # pure variable selection
    mu_i = np.mean(data, axis=0)
    mean_data = np.mean(data, axis=0)
    sigma_i = np.std(data, axis=0)
    max_mean_alpha = np.max(mu_i * alpha)
    purity = sigma_i / (mu_i + max_mean_alpha)
    d_norm = np.sqrt((mu_i**2 + (sigma_i + max_mean_alpha) ** 2))
    D = np.divide(data, d_norm)
    S = np.zeros((comp, c))

    # start loop
    for i in range(comp):
        pmax = -1e99

        for j in range(c):
            for k in range(i+1):
                z = np.sum(data[:, j] * data[:, int(ipure[k])]) / (
                    d_norm[j] * d_norm[int(ipure[k])]
                )
                coo[k, i] = z
                coo[i, k] = z

            coo[i, i] = np.sum(data[:, j] ** 2) / (d_norm[j] ** 2)

            if coo[i, i] > 1e-6:
                var_sim[j] = np.linalg.det(coo[0 : i + 1, 0 : i + 1])
            else:
                var_sim[j] = 0

            if var_sim[j] * purity[j] > pmax:
                pmax = var_sim[j] * purity[j]
                ipure[i] = j

            S[i, j] = sigma_i[j] * var_sim[j]

        w1 = lambda_i**2 / (mean_data**2 + (sigma_i + alpha) ** 2)
        s1 = sigma_i * w1

        for k in range(i):
            z = np.sum(data[:, int(ipure[i])] * data[:, int(ipure[k])]) / (
                d_norm[int(ipure[i])] * d_norm[int(ipure[k])]
            )
            coo[k, i] = z
            coo[i, k] = z

        temp1 = np.sum(data[:, int(ipure[i])] ** 2)
        temp2 = (d_norm[int(ipure[i])])**2
        coo[i, i] = temp1 / temp2


    C = data[:,[int(i) for i in ipure]]
    S = data.T @ C @ np.linalg.inv(C.T @ C)
    P = S.T

    return {"Cinit": C, "Pinit": P}
