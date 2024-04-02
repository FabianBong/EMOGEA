import numpy as np
from .compute_gene_profiles import compute_gene_profiles_func
from EMOGEA.simplisma import simplisma
import pandas as pd
import time

def multivariate_curve_resolution(
    expression_matrix: pd.DataFrame,
    residual_matrix: pd.DataFrame=None,
    number_of_components : int =15,
    init_algorithm : str ="simplisma",
    max_iterations : int =2000,
    tolerance : float =0.0009,
    random_seed : int =1,
    compute_gene_profiles : bool =True,
    verbose : bool =True,
) -> [pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    """
    Multivariate curve resolution algorithm for curve resolution of expression data.

    Parameters
    ----------
    expression_matrix : pd.DataFrame
        A pandas dataframe with genes as rows and samples as columns.

    residual_matrix : pd.DataFrame
        A pandas dataframe with genes as rows and samples as columns.

    number_of_components : int
        Number of components to project onto.

    init_algorithm : str
        Initialization algorithm.

    max_iterations : int
        Maximum number of iterations.

    tolerance : float
        Tolerance for convergence.

    random_seed : int
        Random seed.

    compute_gene_profiles : bool
        Compute gene profiles.

    verbose : bool
        Print progress.

    Returns
    -------
    P : pd.DataFrame
        The profile matrix.
    C : pd.DataFrame
        The contribution matrix.
    Xcalc : pd.DataFrame
        The estimated matrix.
    gene_profiles : pd.DataFrame
        The gene profiles.
    """

    # get start time
    start_time = time.time()

    X = expression_matrix.to_numpy()
    Xres = residual_matrix.to_numpy() if residual_matrix is not None else None
    ncomp = number_of_components
    maxiter = max_iterations

    # set algorithm based on residual matrix
    alg = "weighted" if residual_matrix is not None else "nonweighted"

    # get n and m
    n, m = X.shape

    tol = tolerance
    flg = 0

    # transpose expression and residual matrix
    Xoriginal = X
    X = X.T
    X = X
    Xres = Xres.T if Xres is not None else None

    # initialize P
    if init_algorithm == "simplisma":
        C,P = simplisma(expression_matrix, number_of_components=ncomp)
        Prof = P
    else:
        np.random.seed(random_seed)
        itmp = np.random.choice(np.arange(m), size=m)
        Prof = X[itmp[0:ncomp], :]
        Prof[Prof < 0] = 0

    # define euclidian distance function
    def euclidian_distance(a):
        return np.sqrt(np.sum((a - np.mean(a)) ** 2))

    # define the root mean square deviation
    def rmse(actual, predicted):
        return np.sqrt(np.mean((actual - predicted))** 2)

    # Normalize Prof to unit length
    for i in range(ncomp):
        Prof[i, :] = Prof[i, :] / euclidian_distance(Prof[i, :])

    # Aletnrating TLS
    P = Prof

    # if alg is weighted
    if alg == "weighted":

        # init
        Xwt = 1 / (Xres**2)
        icnt = 0.0009
        rmsdif = 1

        # loop
        while rmsdif > tol and icnt < maxiter:

            Pold = P
            icnt = icnt + 1

            # display progress if verbose
            print("Iteration: ", icnt) if verbose else None

            # init matrix of calutlaed data
            Xcalc = np.zeros((m, n))

            # loop
            for i in range(m):
                values = Xwt[i, :]
                QQ = np.diag(values)
                Xtrunc = P @ QQ @ P.T
                Xstrt = X[i, :] @ QQ @ P.T

                # projected
                Xcalc[i, :] = Xstrt @ np.linalg.inv(Xtrunc) @ P

            # solve for C using projected def
            C = Xcalc @ P.T @ np.linalg.inv(P @ P.T)
            C[C < 0] = 0.0
            Cold = C

            # normalize
            for i in range(ncomp):
                C[:, i] = C[:, i] / euclidian_distance(C[:, i])

            Xcalc = np.zeros((m, n))

            for i in range(n):
                QQ = np.diag(Xwt[:, i])
                Xcalc[:,i] = C @ np.linalg.inv(C.T @ QQ @ C) @ (C.T @ (QQ @ X[:,i]))

            # solve for P using projected data
            P = np.linalg.inv(C.T @ C) @ C.T @ Xcalc
            P[P < 0] = 0.0

            # normalize
            for i in range(ncomp):
                P[i, :] = P[i, :] / euclidian_distance(P[i, :])

            # calculate rms differences
            rms = rmse(P, Pold)
            

            if rms > rmsdif:
                print("Warning possible local minimum") if verbose else None
                print("Consider different initial conditions") if verbose else None
                break


            rmsdif = rms
            print("Converging to: ", tol, "Current: ", rmsdif) if verbose else None

            if rmsdif < tol:
                print("Converged") if verbose else None
                break

            if icnt == maxiter:
                print("Maximum iterations reached") if verbose else None
                break

        Xcalc = np.zeros((m,n))

        # max likelihood
        for i in range(m):
            values = Xwt[i, :]
            QQ = np.diag(values)
            Xtrunc = P @ QQ @ P.T
            Xstrt = X[i, :] @ QQ @ P.T

            # projected
            Xcalc[i, :] = Xstrt @ np.linalg.inv(Xtrunc) @ P


        C = Xcalc @ P.T @ np.linalg.inv(P @ P.T)
        C[C < 0] = 0.0

        # transopes Xcalc
        Xcalc = Xcalc.T

        # set rownames and colnames
        Xcalc = pd.DataFrame(Xcalc, index=expression_matrix.index, columns=expression_matrix.columns)


            
    # if alg is nonweighted
    if alg == "nonweighted":
        # init
        icnt = 0
        rmsdif = 1

        # loop
        while rmsdif > tol and icnt < maxiter:
            Pold = P
            icnt = icnt = 1

            # display progress if verbose
            if verbose:
                print("Iteration: ", icnt)

            # solve for C using projected data
            C = X @ P.T @ np.linalg.inv(P @ P.T)
            C[C < 0] = 0
            Cold = C

            # loop to normalize vectors in contribution matrix
            for i in range(ncomp):
                C[:,i] = C[:,i] / euclidian_distance(C[:,i])
            #C /= np.linalg.norm(C, axis=0)

            # solve for po using contribution matrix and projected data
            P = np.linalg.inv(C.T @ C) @ C.T @ X
            P[P < 0] = 0

            # normalize P
            for i in range(ncomp):
                P[i, :] = P[i, :] / euclidian_distance(P[i, :])
            #P = P / np.linalg.norm(P, axis=1)[:, np.newaxis]

            # calculate rmsdif and update icnt
            rmsdif = rmse(P, Pold)

            # calculate rms differences
            if rmsdif < tol or icnt == maxiter:
                print("Converged") if verbose else None
                break

            # check to see if max iterations has been reached
            if icnt == maxiter:
                print("Maximum iterations reached") if verbose else None
                break

        Xcalc = (C @ P).T

        # set row names and column names
        Xcalc = pd.DataFrame(Xcalc, index=expression_matrix.index, columns=expression_matrix.columns)


    # convert P and C to dataframes
    P = pd.DataFrame(P, columns=expression_matrix.index, index=range(1, ncomp + 1))
    C = pd.DataFrame(C, index=expression_matrix.columns, columns=range(1, ncomp + 1))

    gene_profiles = None
    if compute_gene_profiles:
        print("Computing gene profiles") if verbose else None
        gene_profiles = compute_gene_profiles_func(expression_matrix, C)

    # print execution time
    print("Execution time (in seconds): ", time.time() - start_time) if verbose else None

    # create output list
    return P, C, Xcalc, gene_profiles
