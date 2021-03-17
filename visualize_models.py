import tensorflow as tf
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import glob, os
from datetime import datetime



params = {
    'dataset': "/Users/christianbv/PycharmProjects/EiT/final_datasets",
    "train_split": 0.7,
    "val_split": 0.2,
    "test_split": 0.1
}

MODEL_TYPE = "rnn_updated"
LOG_DIR = f"/Users/christianbv/PycharmProjects/EiT/tmp/logs/{MODEL_TYPE}"
LOG_LEVEL = "ERROR"
TARGET_PATH = "/Users/christianbv/PycharmProjects/EiT/final_datasets"

model_24h_path_local = "/Users/christianbv/PycharmProjects/EiT/tmp/models/rnn_updated/rnn_updated_20210303_162009/model"
model_48h_path_local = "/Users/christianbv/PycharmProjects/EiT/tmp/models/rnn_updated/rnn_updated_20210303_172934/model"

model_24h = tf.keras.models.load_model(f"{model_24h_path_local}/saved_model")
model_48h = tf.keras.models.load_model(f"{model_48h_path_local}/saved_model")


# Load all .parquet files as dataframes
dfs = {}
for path in glob.glob(f"{TARGET_PATH}/**/*.parquet", recursive=True):
    df = pd.read_parquet(path)
    df_name = path.split(os.sep)[-1].split('.')[0]
    dfs[df_name] = df

# Convert dataframe to numpy arrays
train_df, val_df, test_df = dfs['train'], dfs['val'], dfs['test']

train_data = train_df.values.astype('float32')
train_targets = train_df['nox'].values.astype('float32')

val_data = val_df.values.astype('float32')
val_targets = val_df['nox'].values.astype('float32')

test_data = test_df.values.astype('float32')
test_targets = test_df['nox'].values.astype('float32')

data_arrays = train_data, val_data, test_data
target_arrays = train_targets, val_targets, test_targets

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

params = {
    'input_sequence_len': 48,
    'output_sequence_len': 24,
}

# Get data from dataframes
dates = val_df.index.values
data = val_df.values.astype('float32')
labels = val_df['nox'].values.astype('float32')

def eval_predictions(start_idx, IN_SIZE, OUT_SIZE, input_delay=16):
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
    return baseline_mse, model_mse, baseline_mse - model_mse


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
    data_info = pd.read_csv("/Users/christianbv/PycharmProjects/EiT/tmp/data_info/model_summary.txt")
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
    fig.savefig(f"/Users/christianbv/PycharmProjects/EiT/plots/lstm_prediction_{datetime.now():%Y%m%d_%H%M%S}.png", transparent=False, dpi=288)
    fig.show()

IN_SIZE = params['input_sequence_len']
OUT_SIZE = params['output_sequence_len']

for start_idx in range(100, 125, 24):
    print(start_idx)
    plot_predictions(start_idx, IN_SIZE, OUT_SIZE)
    baseline_mse, model_mse, diff_mse = eval_predictions(start_idx, IN_SIZE, OUT_SIZE)
    print(f"Baseline MSE: {baseline_mse}, model MSE: {model_mse}, diff: {diff_mse}")