import pandas as pd


train_dataset_path = 'datasets/fraudTrain.csv'
test_dataset_path = 'datasets/fraudTest.csv'

def load_train_dataset():
    return _load_dataset(train_dataset_path)

def load_test_dataset():
    return _load_dataset(test_dataset_path)

def _load_dataset(path):
    df = pd.read_csv(path, index_col=0)
    df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
    df['dob'] = pd.to_datetime(df['dob'])
    return df.drop(['is_fraud'], axis=0), df['is_fraud']