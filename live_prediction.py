import tensorflow as tf
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import glob, os
from live_data import lastX
from live_data import getLastXHours
from datetime import datetime



MODEL_TYPE = "rnn_updated"
LOG_DIR = f"../EiT/tmp/logs/{MODEL_TYPE}"
LOG_LEVEL = "ERROR"
TARGET_PATH = "../EiT/final_datasets"

model_24h_path_local = "../EiT/tmp/models/rnn_updated/rnn_updated_20210303_162009/model"

model_24h = tf.keras.models.load_model(f"{model_24h_path_local}/saved_model")


params = {
    'input_sequence_len': 48,
    'output_sequence_len': 24,
}


"""
    Returns the live data, i.e. downloads last 48h of data from the various providers and returns it
    Returns: 
        dates   : pd.DataFrame.index.values                     -> The dates of the input data
        data    : pd.DataFrame.values.astype('float32')         -> The input data that is given to the model
        labels  : pd.DataFrame.values.astype('float32')         -> The labels of the input data given to the model
"""
def get_live_data():
    pass

"""
    Performs prediction on the given data
    Assumes data is on the same format as the data returned from "get_live_data"
    Returns:
        pred_dates         : pd.DataFrame.index.values                     -> The dates of the prediction interval (24h)
        model_labels       : pd.DataFrame.values.astype('float32')         -> The predicted values of the model
        baseline_labels    : pd.DataFrame.values.astype('float32')         -> The predicted values of the baseline
"""
def perform_prediction(dates, data, labels):
    pass


"""
    Takes in data, parses into correct format
"""
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

#Build time series dataset
def build_datasets(data_arrays, target_arrays, params, hparams):
    def build_dataset(data, sequence_length=24, batch_size=32):
        return tf.keras.preprocessing.timeseries_dataset_from_array(
            data, None, sequence_length=sequence_length,
            sampling_rate=1, batch_size=batch_size
        )
    all_params = {**params, **hparams}
    batch_size = all_params['batch_size']

    # Build TensorFlow Datasets
    train_dataset_data = build_dataset(train_data[:-all_params['output_sequence_len'], ...], sequence_length=all_params['input_sequence_len'], batch_size=batch_size)
    val_dataset_data = build_dataset(val_data[:-all_params['output_sequence_len'], ...], sequence_length=all_params['input_sequence_len'], batch_size=batch_size)
    test_dataset_data = build_dataset(test_data[:-all_params['output_sequence_len'], ...], sequence_length=all_params['input_sequence_len'], batch_size=batch_size)

    train_dataset_targets = build_dataset(train_targets[all_params['input_sequence_len']:], sequence_length=all_params['output_sequence_len'], batch_size=batch_size)
    val_dataset_targets = build_dataset(val_targets[all_params['input_sequence_len']:], sequence_length=all_params['output_sequence_len'], batch_size=batch_size)
    test_dataset_targets = build_dataset(test_targets[all_params['input_sequence_len']:], sequence_length=all_params['output_sequence_len'], batch_size=batch_size)

    train_dataset = tf.data.Dataset.zip((train_dataset_data, train_dataset_targets))
    val_dataset = tf.data.Dataset.zip((val_dataset_data, val_dataset_targets))
    test_dataset = tf.data.Dataset.zip((test_dataset_data, test_dataset_targets))
    return train_dataset, val_dataset, test_dataset

#Baseline model
class BaselineModel(tf.keras.Model):
    def __init__(self, params, hparams, **kwargs):
        super().__init__(**kwargs)
        all_params = {**params , **hparams}
        self.output_sequence_len = all_params['output_sequence_len']

    def call(self, inputs):
        raw_nox = inputs[..., 0] # Moved NOX to first column in dataframe
        raw_labels = raw_nox[..., -1]
        repeated_labels = tf.repeat(raw_labels[..., None], self.output_sequence_len, axis=1)
        return repeated_labels


# Converts into data_arrays

# Perform prediction and generate plot
def plot_predictions(start_idx, IN_SIZE, OUT_SIZE, input_delay=16):
    # start_idx = 1348
    end_idx = start_idx + (IN_SIZE + OUT_SIZE) + 2 * input_delay

    dates_slice = dates[start_idx:end_idx]
    data_slice = data[start_idx:end_idx]
    labels_slice = labels[start_idx:end_idx]

    # Create input window
    input_start_idx = input_delay
    input_end_idx = input_delay + IN_SIZE

    input_dates = dates_slice[input_start_idx:input_end_idx]
    input_data = data_slice[input_start_idx:input_end_idx]
    input_labels = labels_slice[input_start_idx:input_end_idx]

    # Create prediction window
    pred_start_idx = input_end_idx
    pred_end_idx = pred_start_idx + OUT_SIZE

    pred_dates = dates_slice[pred_start_idx:pred_end_idx]
    pred_data = data_slice[pred_start_idx:pred_end_idx]
    pred_labels = labels_slice[pred_start_idx:pred_end_idx]

    # Model predictions
    batched_input = input_data.reshape(1, IN_SIZE, -1)
    batched_labels = pred_labels.reshape(1, OUT_SIZE)
    model_labels = model_24h.predict(batched_input)
    model_mse = model_24h.evaluate(batched_input, batched_labels, return_dict=True, verbose=0)['loss']
    # print(model_mse)

    # Baseline predictions
    baseline_model = BaselineModel(params=params, hparams={})
    baseline_model.compile(loss=tf.keras.losses.MeanSquaredError())
    baseline_labels = baseline_model.predict(batched_input)
    baseline_mse = baseline_model.evaluate(batched_input, batched_labels, return_dict=True, verbose=0)['loss']
    # print(baseline_mse)

    # Denormalizing values, i.e. multiplying with stddeviation and adding the mean
    data_info = pd.read_csv("../EiT/tmp/data_info/model_summary.txt")
    old_nox_vals = data_info[data_info['variable'] == 'nox']
    mean, std = old_nox_vals['mean'].values[0], old_nox_vals['std'].values[0]

    baseline_labels = baseline_labels * std + mean
    model_labels = model_labels * std + mean
    labels_slice = labels_slice * std + mean
    input_labels = input_labels * std + mean


    # Plot results
    sns.set_style("darkgrid")

    fig = plt.figure(figsize=(14, 5))
    # plt.title(f"Baseline: {baseline_mse}   Model: {model_mse}   Diff: {baseline_mse - model_mse}")
    sns.lineplot(x=dates_slice, y=labels_slice, color="#202020", linewidth=0.4)
    sns.lineplot(x=input_dates, y=input_labels, color="green", linewidth=0.8)
    # sns.lineplot(x=pred_dates, y=pred_labels, color="blue", style=True, dashes=[(2,2)])
    sns.lineplot(x=pred_dates, y=model_labels.squeeze(), color="#802020", style=True, dashes=[(4, 4)], linewidth=1.4)
    sns.lineplot(x=pred_dates, y=baseline_labels.squeeze(), color="#202020", style=True, dashes=[(4, 4)], linewidth=1.4)

    plt.axvline(input_dates[0], color="#202020", linewidth=0.5)
    plt.axvline(input_dates[-1], color="#202020", linewidth=0.5)

    plt.axvline(pred_dates[0], color="#202020", linewidth=0.5)
    plt.axvline(pred_dates[-1], color="#202020", linewidth=0.5)
    plt.gca().set_xticks(dates_slice[::6])

    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    # plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    plt.gca().legend(labels=('NOx', 'Input window', 'Predictions (24h)', 'Baseline'), ncol=1,
                     loc="upper left", frameon=True)

    plt.gca().set(ylabel="NOx", xlabel="Timestamp")
    # rotate and align the tick labels so they look better
    fig.autofmt_xdate()

    # plt.xticks(rotation=45)
    # plt.gca().set_xticklabels(f"{dates_slice[::3]:%Y-%m-%d %H:%M}", rotation=45)
    fig.tight_layout()
    fig.savefig(f"../EiT/plots/lstm_prediction_{datetime.now():%Y%m%d_%H%M%S}.png", transparent=False, dpi=288)
    fig.show()