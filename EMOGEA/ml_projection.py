import numpy as np
import pandas as pd


def ml_projection(
    expression_matrix: pd.DataFrame,
    error_covariance_matrix: pd.DataFrame,
    number_of_components: int = 15,
    max_iterations: int = 2000,
    tolerance: float = 1e-6,
    verbose: bool = True,
) -> dict:
    """
    Maximum likelihood projection of expression matrix onto a lower dimensional space.

    Parameters
    ----------
    expression_matrix : pd.DataFrame
        A pandas dataframe with genes as rows and samples as columns.

    error_covariance_matrix : pd.DataFrame
        A pandas dataframe with genes as rows and samples as columns.

    number_of_components : int
        Number of components to project onto.

    max_iterations : int
        Maximum number of iterations.

    tolerance : float
        Tolerance for convergence.

    verbose : bool
        Print progress.

    Returns
    -------
    U : pd.DataFrame
        The U matrix.
    S : pd.DataFrame
        The S matrix.
    V : pd.DataFrame
        The V matrix.
    estimated_matrix : pd.DataFrame
        The estimated matrix.
    """

    # assign to variables
    X = expression_matrix.to_numpy()
    Xcov = error_covariance_matrix.to_numpy()
    ncomp = number_of_components

    # Transpose the expression matrix
    X = X.T

    # set ML parameters
    pc = ncomp
    flg = 1
    maxiter = max_iterations
    lam = tolerance
    iter = 0

    # compute svd
    U, S, V = np.linalg.svd(X, full_matrices=False)
    S = np.diag(S)

    # select V with the first pc columns
    V = V[:pc, :].T

    # now do maximum likelihood
    while flg == 1 and iter < maxiter:
        iter = iter + 1

        # estimate matrix
        Xhat = X @ (Xcov) @ V @ np.linalg.inv(V.T @ (Xcov) @ V) @ V.T
        S1 = np.sum(np.diag((X - Xhat) @ (Xcov) @ (X - Xhat).T))

        # improve
        U, S, V = np.linalg.svd(Xhat, full_matrices=False)
        Xhat = U[:, :pc] @ (U[:, :pc]).T @ X
        S2 = np.sum(np.diag((X - Xhat) @ (Xcov) @ (X - Xhat).T))
        U, S, V = np.linalg.svd(Xhat, full_matrices=False)
        U = U[:, :pc]
        S = np.diag(S[:pc])
        V = V[:pc, :].T

        # calculate objective
        sobj = np.abs((S1 - S2) / S2)
        if sobj < lam:
            flg = 0

        if verbose:
            print("Iteration: ", iter, "Objective: ", sobj, "Tolerance: ", lam)

    # compute estimated values
    estimated_matrix = (U @ S @ V.T).T
    
    # set colnames 
    estimated_matrix = pd.DataFrame(estimated_matrix, index=expression_matrix.index, columns=expression_matrix.columns)

    # make U, S and V dataframes with row/column names
    U = pd.DataFrame(U, index=expression_matrix.columns, columns=range(1, pc + 1))
    S = pd.DataFrame(S, index=range(1, pc + 1), columns=range(1, pc + 1))
    V = pd.DataFrame(V, index=expression_matrix.index, columns=range(1, pc + 1))

    # retrun
    return U,S,V, estimated_matrix
