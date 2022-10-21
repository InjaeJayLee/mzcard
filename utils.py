from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Tuple
import pandas as pd


def change_column_name(df: pd.DataFrame, from_to: Tuple[str, str]):
    old_name, new_name = from_to
    df.rename(columns={old_name: new_name}, inplace=True)

def convert_category_to_num_label(df: pd.DataFrame, col_name: str):
    encoder = LabelEncoder()
    store_label = encoder.fit_transform(df[col_name])
    df[col_name] = store_label


def draw_heatmap(df: pd.DataFrame):
    plt.figure(figsize=(30, 30))
    sns.set(font_scale=2.5)
    sns.heatmap(df.corr(), annot=True, annot_kws={"size": 24})
    plt.show()