import pandas as pd

def load_sample() -> [pd.DataFrame, pd.DataFrame]:

	"""
	Load sample data from csv files

	Returns
	-------
	data: pd.DataFrame
		Expression data

	meta_data: pd.DataFrame
		Meta data
	"""

	df = pd.read_csv("EMOGEA/data/expressionData.csv", index_col=1)

	# drop first column
	df = df.drop(df.columns[0], axis=1)

	# rename first index name to "Gene"
	df.index.names = ["Gene"]

	# load meta data
	meta_data = pd.read_csv("EMOGEA/data/metaData.csv", index_col=0)

	return df,meta_data

