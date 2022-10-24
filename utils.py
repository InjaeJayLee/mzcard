from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Tuple
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor


########## Preprocessing ##########
onehot_encoders = {}
def change_column_name(df: pd.DataFrame, from_to: Tuple[str, str]):
    old_name, new_name = from_to
    df.rename(columns={old_name: new_name}, inplace=True)

def convert_category_to_num_label(df: pd.DataFrame, col_name: str):
    encoder = LabelEncoder()
    store_label = encoder.fit_transform(df[col_name])
    df[col_name] = store_label

def get_new_df_with_onehot_encoding(df: pd.DataFrame, col_name: str, is_train_data: bool):
    """
        params:
            df: if this is true, return pandas dataframe, otherwise numpy 2D array
            col_name: the name of a column to convert
            is_train_data: whether or not it is train data

        return: a new data frame with the onehot encoding
    """
    encoder = OneHotEncoder(sparse=False)
    if is_train_data:
        cat = encoder.fit_transform(df[[col_name]])
        new_cols = pd.DataFrame(cat, columns=[col_name + "_" + col for col in encoder.categories_[0]])
        new_df = pd.concat([df.drop(columns=[col_name]), new_cols], axis=1)
        onehot_encoders[col_name] = encoder
        return new_df
    else:
        assert(onehot_encoders.get(col_name))
        encoder = onehot_encoders[col_name]
        cat = encoder.transform(df[[col_name]])
        new_cols = pd.DataFrame(cat, columns=[col_name + "_" + col for col in encoder.categories_[0]])
        new_df = pd.concat([df.drop(columns=[col_name]), new_cols], axis=1)
        return new_df
###################################



########## Statistics #############


###################################



########## Visualization ##########
def draw_heatmap(df: pd.DataFrame):
    """
        params
            df: a data frame to draw a heatmap of
    """
    plt.figure(figsize=(30, 30))
    sns.set(font_scale=2.5)
    sns.heatmap(df.corr(), annot=True, annot_kws={"size": 24})
    plt.show()
###################################