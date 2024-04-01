import pandas as pd
import numpy as np


def compute_gene_profiles_func(
    expression_matrix: pd.DataFrame,
    C: pd.DataFrame,
):

    # define cosine similarity function
    def cosine_similarity(a, b):
        return np.sum(a * b) / np.sqrt(np.sum(a**2) * np.sum(b**2))


    Xorg = expression_matrix.T.to_numpy()
    ncomp = C.shape[1]


    ngenes = Xorg.shape[1]
    angles = np.zeros((ncomp, ngenes))
    for i in range(ncomp):
        for j in range(ngenes):
            angles[i, j] = cosine_similarity(C.iloc[:, i], Xorg[:, j])

    # set colnames of angles 
    angles = pd.DataFrame(angles, columns=expression_matrix.index)

    # remvoe any rows that contain nan
    angles = angles.dropna()

    # Generate the gene profiles
    gene_profiles = pd.DataFrame(np.zeros((angles.shape[1], ncomp)), columns=C.columns, dtype=object)

    for i in range(ncomp):
        # sort the angles by row and get index
        k = np.argsort(angles.iloc[i, :])[::-1]
        gene_profiles.iloc[:, i] = angles.columns[k]    # generate gene profiles 

    return gene_profiles
