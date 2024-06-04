import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


def correlation(df: pd.DataFrame, threshold: float):
    col_corr = set()
    corr_matrix =df.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold:
              colname = corr_matrix.columns[i]
              col_corr.add(colname)
    return col_corr


def load_dataframes(paths: list[str]) -> pd.DataFrame:
    dfs = [pd.read_csv(path) for path in paths]
    df = pd.concat(dfs)
    df.reset_index(drop=True, inplace=True)
    return df  


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    df =  df.drop_duplicates(keep="first")
    df.dropna(inplace=True)

    # change types
    integer = []
    f = []
    for i in df.columns[:-1]:
        if df[i].dtype == "int64": integer.append(i)
        else : f.append(i)

    df[integer] = df[integer].astype("int32")
    df[f] = df[f].astype("float32")

    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    le = LabelEncoder()
    objList = df.select_dtypes(include = "object").columns
    for feat in objList:
        df[feat] = le.fit_transform(df[feat].astype(str))

    # corr_features = correlation(df, 0.85)
    # df.drop(corr_features,axis=1,inplace=True)

    df = df.groupby(' Label').filter(lambda x:len(x)>10000)
    return df


def load_dataset(paths: list[str], preprocess: bool = False, sample_size: float = 1) -> pd.DataFrame:
    df = load_dataframes(paths=paths)
    if sample_size != -1:
        df = df.sample(frac=sample_size)
    if preprocess:
        df = preprocess_data(df=df)
    return df


def get_paths(
        num_of_clients: int, 
        client_num: int, 
        base_path: str = "processed_data/data"
    ) -> list[str]:
    df_pathes = [os.path.join(base_path, path)
                for path in os.listdir(base_path)]    
    step = int(len(df_pathes) / num_of_clients)    
    return df_pathes[client_num*step:(client_num+1)*step]


def split_and_save_data_frame(
        df: pd.DataFrame,
        output_path: str,
        file_counts: int
    ):
    len_df = len(df.index)
    step = int(len_df / (file_counts))
    for i in range(file_counts):
        df[i*step:(i+1)*step].to_csv(f"{output_path}/train{i}.csv", index=False)


def get_x_y(
    df: pd.DataFrame,
    binary: bool = True,
    reshape: bool = True
):
    if binary:
        df['classify_label'] = df.apply(
            lambda row: 1 if row[" Label"] > 0 else 0, 
            axis=1
        )
    else:
        labels = sorted(df[" Label"].unique())
        df['classify_label'] = df.apply(
            lambda row: labels.index(row[" Label"]),
            axis=1
        )

    X = df.drop([' Label', "classify_label"], axis=1)
    Y = df['classify_label']
    scaler = RobustScaler()
    scaler.fit_transform(X)

    x_test = X.values
    y_test = Y.values

    if reshape:
        # Reshape x_test to have a time_steps dimension
        time_steps = 1
        x_test = x_test.reshape((x_test.shape[0], time_steps, x_test.shape[1]))
    return x_test, y_test


if __name__ == "__main__":
    paths = get_paths(1, 0, "processed_data_3")
    df = load_dataset(paths=paths)
    
    print(len(df))

    # print(sorted(df[" Label"].unique()))
    # df = df.sample(frac=1).reset_index(drop=True)
    # train, test = train_test_split(df, random_state=42, test_size=0.1)
    # test.to_csv("processed_data_3/test.csv", index=False)
    # split_and_save_data_frame(
    #     df=train,
    #     output_path="processed_data_3/",
    #     file_counts=8
    # )