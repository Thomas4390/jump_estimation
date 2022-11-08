import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def convert_to_datetime(df: pd.DataFrame) -> pd.DataFrame:
    df["Date"] = pd.to_datetime(df["Date"])

    return df


# Convert data from xlsx format to pickle format


def convert_data(filename: str) -> None:
    df_excel = pd.read_excel(filename)
    df_excel = convert_to_datetime(df_excel)
    # Fill all missing values backwards
    df_excel = df_excel.fillna(method="bfill")
    # Drop any remaining missing values
    df_excel = df_excel.dropna()
    df_excel.to_pickle(f"{filename[0:4]}.pkl")

    return None


def from_excel_to_pkl() -> None:

    file_to_convert = ["1990.xlsx", "2001.xlsx", "2007.xlsx", "2018.xlsx"]

    for file in file_to_convert:
        convert_data(file)

    return None


# Read data from pickle format


def concat_data() -> pd.DataFrame:
    df_2001 = pd.read_pickle("images/data/2001.pkl")
    df_1990 = pd.read_pickle("images/data/1990.pkl")
    df_2007 = pd.read_pickle("images/data/2007.pkl")
    df_2018 = pd.read_pickle("images/data/2018.pkl")

    df_concat = pd.concat([df_1990, df_2001, df_2007, df_2018], ignore_index=True)

    return df_concat


def save_to_pickle(df: pd.DataFrame, filename: str) -> None:
    df.to_pickle(f"{filename}.pkl")

    return None


if __name__ == "__main__":
    from_excel_to_pkl()
    df_full = concat_data()
    save_to_pickle(df_full, "full_data")
