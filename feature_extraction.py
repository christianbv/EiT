from datetime import datetime
import numpy as np
import pandas as pd
import os
from contextlib import redirect_stdout


params = {
    "train_split": 0.7,
    "val_split": 0.2,
    "test_split": 0.1,
    "save_path": "/Users/christianbv/PycharmProjects/EiT/final_datasets/",
    "load_path": "/Users/christianbv/PycharmProjects/EiT/raw_datasets/danmarks_plass_v0.csv"
}

def load_file(path = params['load_path']):
    df = pd.read_csv(path,
                     engine = 'python',
                     encoding = 'utf-16',
                     sep = '\t',
                     skiprows= 1,
                     names = ["Date",
                              "MET_air_temperature_2m",
                              "MET_relative_humidity",
                              "MET_surface_air_pressure",
                              "MET_wind_from_direction_10m",
                              "MET_wind_speed_10m", "MET_precipation",
                              "Nea_No", "Nea_No2","NEA_NOx", "Nea Pm2_5","Nea_Pm10",
                              "PRA_1__upTo5_6", "PRA_3__upTo5_6", "PRA_4__upTo5_6",
                              "PRA_7__upTo5_6", "PRA_8__upTo5_6", "PRA_9__upTo5_6",
                              "PRA_10__upTo5_6", "PRA_11__upTo5_6", "PRA_12__upTo5_6",
                              "PRA_13__upTo5_6",
                              "MET_precipation_last_24h",
                              "MET_wind speed_last_24h"
                              ])
    return df


def parse_file(df):
    columns = {
        "MET_relative_humidity": ["hum", "Relative humidity at the location"],
        "MET_surface_air_pressure": ["hpa", "Air pressure at the surface"],
        "MET_air_temperature_2m": ["temp", "Air temperature at 2m height over the location"],
        "MET_wind_speed_10m": ["wind", "Wind speed at 10m for the location"],
        "MET_wind_from_direction_10m": ["wind_dir", "Wind direction"],
        "NEA_NOx": ["nox", "The label we want to predict"],
        "PRA_tonnes_vehicles_passing": ["traffic",
                                        "Weighted feature saying something about the length of the vehicles passing by"],
        "location": ["location", "_"],
        "Date": ["date", "_"],
    }
    df.reset_index(inplace=True)
    df = df.rename({k: v[0] for k, v in columns.items()}, axis=1)
    for col in df.columns:
        if col == 'date' or col == 'index':
            continue
        if 'PRA' in col.split('_'):
            df[col] = pd.to_numeric(df[col], downcast='integer')
        else:
            df[col] = df[col].astype(str).str.replace(',','.').replace('nan',np.NAN)
            df[col] = pd.to_numeric(df[col],downcast='float')
    df['date'] = pd.to_datetime(df['date'].astype(str), format='%d/%m/%Y %H:%M:%S').round('H')

    df.set_index("date", inplace=True)
    df.sort_index(axis=0, inplace=True)
    # Converting wind direction and wind strength into wind vectors
    wind = df["wind"]
    wind_rad = df["wind_dir"] * np.pi / 180
    df['windX'] = wind * np.cos(wind_rad)
    df['windY'] = wind * np.sin(wind_rad)
    df.drop(["wind", "wind_dir", 'index'], axis=1, inplace=True)

    # Replacing all null values with mean
    df.fillna(df.mean(), inplace = True)

    # Splitting up the index dates into own quarter feature:
    df['quarter'] = df.index.quarter
    # df = df[(df['quarter'] == 1) | (df['quarter'] == 2)]

    from_ts = "2015-07-01"
    to_ts = "2019-06-30"
    df = df[from_ts:to_ts].copy()

    # Add periodic features to help with seasonality
    timestamp_s = df.index.map(datetime.timestamp)
    day = 24 * 60 * 60
    year = (365.2425) * day
    week = 7 * day
    df['day_sin'] = np.sin(timestamp_s * (2 * np.pi / day))
    df['day_cos'] = np.cos(timestamp_s * (2 * np.pi / day))
    df['week_sin'] = np.sin(timestamp_s * (2 * np.pi / week))
    df['week_cos'] = np.cos(timestamp_s * (2 * np.pi / week))
    df['year_sin'] = np.sin(timestamp_s * (2 * np.pi / year))
    df['year_cos'] = np.cos(timestamp_s * (2 * np.pi / year))

    # Put nox in front
    df = df[['nox'] + [col for col in df.columns if col != 'nox']]
    df.sort_index(axis=0, inplace=True)
    return df

def create_sets(df):
    TRAIN_SPLIT = params['train_split']
    VAL_SPLIT = params['val_split']
    TEST_SPLIT = params['test_split']
    split_array = np.array([TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT])

    # Actual and desired class distributions
    total_size = df.shape[0]
    split_indices = np.ceil(split_array.cumsum() * total_size).astype(int)

    # Create dataframes
    train_df = df.iloc[:split_indices[0]]
    val_df = df.iloc[split_indices[0]:split_indices[1]]
    test_df = df.iloc[split_indices[1]:]

    # Normalization
    # Note that we normalize validation and test data with same parameters as training data!!
    train_mean = train_df.mean()
    train_std = train_df.std()

    train_df = (train_df - train_mean) / train_std
    val_df = (val_df - train_mean) / train_std
    test_df = (test_df - train_mean) / train_std

    # Writing train_mean and train_std to a file, for denormalizing
    base_path = "/Users/christianbv/PycharmProjects/EiT/tmp/data_info"
    model_summary_path = os.path.join(base_path, "model_summary.txt")
    data_info = pd.DataFrame(data = zip(train_mean, train_std), columns = ["mean", "std"], index = train_mean.index)
    data_info.to_csv(model_summary_path, index_label="variable")

    return train_df, val_df,test_df

def save_files(train_df, val_df, test_df, save_path = params['save_path']):
    train_df.to_parquet(f"{save_path}train.parquet")
    val_df.to_parquet(f"{save_path}val.parquet")
    test_df.to_parquet(f"{save_path}test.parquet")


df = load_file()
df = parse_file(df)
train_df,val_df,test_df = create_sets(df)
#save_files(train_df,val_df,test_df)
