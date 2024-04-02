import pandas as pd
import numpy as np


def prepare_data(
    expression_data: pd.DataFrame,
    meta_data: pd.DataFrame,
    sample_column: str = "ID",
    condition_column: str = "condition",
    apply_log_transformation: bool = True,
) -> [pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    This function prepares the data to be passed to the ml projection function.

    Parmeters
    ---------
    expression_data: pd.dataframe
        The expression data with samples as columns and genes as rows

    meta_data: pd.DataFrame
        The meta data with samples as rows and conditions as columns

    sample_column: str
        The column in the meta data that contains the sample names

    condition_column: str
        The column in the meta data that contains the condition names

    apply_log_transformation: bool
        Whether to apply a log transformation to the dataframe

    Raises
    ------
    ValueError
        If the condition or sample column is not found in the meta dataframe

    Returns
    -------
    expression_matrix: pd.DataFrame
        The expression matrix
    residual_matrix: pd.DataFrame
        The residual expressionMatrix
    error_covariance_matrix: pd.DataFrame
        The error covariance matrix
    """

    # check if conditioncolum and samplecolumn are in meta_data
    if (
        condition_column not in meta_data.columns
        or sample_column not in meta_data.columns
    ):
        raise ValueError("Condition or sample column not found in meta data")

    # check if number of samples in expression data and meta data are the samples
    if len(expression_data.columns) != len(meta_data[sample_column].unique()):
        raise ValueError(
            "Number of samples in expression data and meta data do not match"
        )

    # redored the columns of the expression data to match the order of the samples in the meta dataframe
    expression_data = expression_data[meta_data[sample_column].unique()]

    # Determine replicates for each condition
    meta_data["replicates"] = 0

    # for each condition
    for condition in meta_data[condition_column].unique():
        # get all samples for this condition
        samples = meta_data[meta_data[condition_column] == condition][sample_column]
        # count the number of samples
        meta_data.loc[meta_data[condition_column] == condition, "replicates"] = range(
            len(samples)
        )

    # convert dataframe to numpy array
    X = expression_data.to_numpy()

    # set values that are 0 to nanX
    X[X == 0] = np.nan

    # apply log transformation
    if apply_log_transformation:
        X = np.log2(X)

    n, m = X.shape
    Xres = np.zeros((n, m))

    # for each condition
    for condition in meta_data[condition_column].unique():
        condition_index = np.where(meta_data[condition_column] == condition)[0]
        #centered_values = scale(X[:, condition_index], with_std=False, axis=0)
        centered_values = X[:, condition_index] - np.nanmean(X[:, condition_index], axis=0)
        Xres[:, condition_index] = centered_values

    # set values that are na in Xres to 999
    Xres[np.isnan(Xres)] = 999
    X[np.isnan(X)] = 0

    # calculate error covarinaceXwt
    Xwt = 1 / Xres
    Xcov = Xwt @ np.transpose(Xwt)

    # convert X, Xres and Xcov to pandas dataframes
    X = pd.DataFrame(X, index=expression_data.index, columns=expression_data.columns)
    Xres = pd.DataFrame(
        Xres, index=expression_data.index, columns=expression_data.columns
    )
    Xcov = pd.DataFrame(
        Xcov, index=expression_data.index, columns=expression_data.index
    )

    # return
    return X, Xres, Xcov
