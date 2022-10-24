import pandas as pd


train_dataset_path = 'datasets/fraudTrain.csv'
test_dataset_path = 'datasets/fraudTest.csv'

def load_train_dataset(is_dataframe=True):
    """
        params:
            is_dataframe: if this is true, return pandas dataframe, otherwise numpy 2D array

        return: pandas dataframe or numpy 2D-array
    """
    return _load_dataset(train_dataset_path, is_dataframe)

def load_test_dataset(is_dataframe=True):
    """
        params:
            is_dataframe: if this is true, return pandas dataframe, otherwise numpy 2D array

        return: pandas dataframe or numpy 2D-array
    """
    return _load_dataset(test_dataset_path, is_dataframe)

def _load_dataset(path, is_dataframe):
    df = pd.read_csv(path, index_col=0)
    df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
    df['dob'] = pd.to_datetime(df['dob'])
    x, y = df.drop(['is_fraud'], axis=1), df['is_fraud']
    if is_dataframe:
        return x, y
    else:
        return x.to_numpy(), y.to_numpy()