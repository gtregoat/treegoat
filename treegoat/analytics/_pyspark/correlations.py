from pyspark.ml.stat import Correlation
from pyspark.ml.feature import VectorAssembler
from pyspark import sql
from pyspark import data
import pandas as pd


def correlation_matrix(df: sql.DataFrame, corr_columns: list, method: str = 'pearson'):
    """

    Args:
        df: pyspark dataframe,
        corr_columns:
        method:

    Returns:

    """
    vector_col = "corr_features"
    assembler = VectorAssembler(inputCols=corr_columns, outputCol=vector_col)
    df_vector = assembler.transform(df).select(vector_col)
    matrix = Correlation.corr(df_vector, vector_col, method)
    result = matrix.collect()[0]["pearson({})".format(vector_col)].values
    return pd.DataFrame(result.reshape(-1, len(corr_columns)), columns=corr_columns, index=corr_columns)

