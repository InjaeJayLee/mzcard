from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import seaborn as sns
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from typing import Tuple, List
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor


########## Preprocessing ##########
onehot_encoders = {}

def convert_category_to_num_label(df: pd.DataFrame, col_name: str):
    """
        params:
            df: pandas dataframe
            col_name: the name of a column to convert

        return: a new data frame with the onehot encoding
    """
    encoder = LabelEncoder()
    store_label = encoder.fit_transform(df[col_name])
    df[col_name] = store_label

def get_new_df_with_onehot_encoding(df: pd.DataFrame, col_name: str, is_train_data: bool):
    """
        params:
            df: pandas dataframe
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
def get_vif(df: pd.DataFrame):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    target_df = df.select_dtypes(include=numerics)
    vif = pd.DataFrame()
    vif["VIF Factor"] = [variance_inflation_factor(target_df.values, i) for i in range(target_df.shape[1])]
    vif["Feature Name"] = target_df.columns
    vif = vif.sort_values("VIF Factor").reset_index(drop=True)
    return vif
###################################



########## Visualization ##########
def draw_heatmap(df: pd.DataFrame):
    """
        params
            df: a pandas dataframe to use to draw a heatmap of it
    """
    plt.figure(figsize=(30, 30))
    sns.set(font_scale=2.2)
    sns.heatmap(df.corr(numeric_only=True), cmap='vlag', annot=True, annot_kws={"size": 24})
    plt.show()

def set_plot_labels(ax, title: Tuple[str, int]=None, xlab: Tuple[str, int]=None, ylab: Tuple[str, int]=None, legend:Tuple[List, int]=None):
    """
        params
            title: title tuple containing a string of its name and its font size
            xlab: x label tuple containing a string of its name and its font size
            ylab: y label tuple containing a string of its name and its font size
            legend: legend tuple containing a string of its name and its font size
    """
    if title:
        plt.title(title[0], fontsize=title[1])
    if xlab:
        ax.set_xlabel(xlab[0], size=xlab[1])
    if ylab:
        ax.set_ylabel(ylab[0], size=ylab[1])
    if legend:
        ax.legend(labels=legend[0], fontsize=legend[1])
###################################