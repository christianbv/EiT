import pandas as pd
import numpy as np
import tensorflow as tf
from tensorboard.plugins.hparams import api as hp
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, LearningRateScheduler
import tensorflow.keras.backend as K
import io, sys, glob, time, os, json
from contextlib import redirect_stdout
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
TARGET_PATH = '/Users/christianbv/PycharmProjects/EiT/final_datasets'

def load_data(TARGET_PATH):
    # Load all .parquet files as dataframes
    dfs = {}
    for path in glob.glob(f"{TARGET_PATH}/**/*.parquet", recursive=True):
        df = pd.read_parquet(path)
        df_name = path.split(os.sep)[-1].split('.')[0]
        dfs[df_name] = df

    # Convert dataframe to numpy arrays
    train_df, val_df, test_df = dfs['train'], dfs['val'], dfs['test']
    return train_df,val_df,test_df

"""
    - Specifies nox as target variable
"""
def create_arrays(train_df, val_df, test_df):
    train_data = train_df.values.astype('float32')
    train_targets = train_df['nox'].values.astype('float32')

    val_data = val_df.values.astype('float32')
    val_targets = val_df['nox'].values.astype('float32')

    test_data = test_df.values.astype('float32')
    test_targets = test_df['nox'].values.astype('float32')

    data_arrays = train_data, val_data, test_data
    target_arrays = train_targets, val_targets, test_targets
    return data_arrays, target_arrays

"""
    Building datasets
"""
def build_datasets(data_arrays, target_arrays, params, hparams):
    train_data, val_data, test_data = data_arrays
    train_targets, val_targets, test_targets = target_arrays

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


"""
    Defining hyperparameters
"""
# Evaluation metrics
metrics = [tf.keras.metrics.MeanSquaredError(name='mse'),
           tf.keras.metrics.MeanAbsoluteError(name='mae'),
           tf.keras.metrics.MeanSquaredLogarithmicError(name='msle')]

hp_metrics = [hp.Metric('mse', display_name='mse'),
              hp.Metric('mae', display_name='mae'),
              hp.Metric('msle', display_name='msle')]

hparams_refs = {
    'recurrent_cell': hp.HParam('recurrent_cell', hp.Discrete(['gru', 'lstm'])),
    'recurrent_layers': hp.HParam('recurrent_layers', hp.IntInterval(2, 4)),
    'recurrent_units': hp.HParam('recurrent_units', hp.Discrete([16, 32, 64, 128])),

    'input_sequence_len': hp.HParam('input_sequence_len', hp.Discrete([24, 48])),
    'output_sequence_len': hp.HParam('output_sequence_len', hp.Discrete([12, 24, 48])),

    'batch_size': hp.HParam('batch_size', hp.Discrete([32, 64, 128, 256])),
    'optimizer': hp.HParam('optimizer', hp.Discrete(['adam', 'sgd'])),
    'loss': hp.HParam('loss', hp.Discrete(['mse', 'mae'])),
    'learning_rate': hp.HParam('learning_rate', hp.Discrete([10 ** -3.5, 10e-4, 10 ** -4.5])),
}

"""
    Build model
"""
cell_types = {
    'gru': tf.keras.layers.GRU,
    'lstm': tf.keras.layers.LSTM,
    'bi-gru': lambda *args, **kwargs: tf.keras.layers.Bidirectional(tf.keras.layers.GRU(*args, **kwargs)),
    'bi-lstm': lambda *args, **kwargs: tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(*args, **kwargs)),
}

optimizers = {
    'adam': tf.keras.optimizers.Adam,
    'rmsprop': tf.keras.optimizers.RMSprop,
    'sgd': tf.keras.optimizers.SGD
}

def build_model(params, hparams):
    all_params = {**params, **hparams}
    model = tf.keras.Sequential()

    # Recurrent units
    for _ in range(hparams['recurrent_layers'] - 1):
        model.add(cell_types[hparams['recurrent_cell']](hparams['recurrent_units'], return_sequences=True,
                                                        kernel_regularizer=tf.keras.regularizers.L2()))
        model.add(tf.keras.layers.Dropout(0.1))
    model.add(cell_types[hparams['recurrent_cell']](hparams['recurrent_units'],
                                                    kernel_regularizer=tf.keras.regularizers.L2()))

    model.add(tf.keras.layers.Dense(all_params['output_sequence_len']))
    return model

def train_model(params, hparams, metrics, log_dir, model_type, **fit_kwargs):
    # Create callbacks and prepare logging
    early_stopping = EarlyStopping(patience=10, restore_best_weights=True, verbose=1)
    timestamp = datetime.now()
    dir_name = f"{model_type}_{timestamp:%Y%m%d_%H%M%S}"
    filename = f"{log_dir}/{dir_name}"
    hparams["__timestamp__"] = int(f"{timestamp:%Y%m%d%H%M%S}")
    tensorboard = TensorBoard(filename, write_graph=False, histogram_freq=0, write_images=False)
    hp_board = hp.KerasCallback(filename, hparams, trial_id=dir_name)
    callbacks = [tensorboard, hp_board, early_stopping]

    # Build model and run
    model = build_model(params, hparams)
    model.compile(
        optimizer=optimizers[hparams['optimizer']](learning_rate=hparams['learning_rate']),
        loss=hparams['loss'],
        metrics=metrics
    )
    history = model.fit(callbacks=callbacks, **fit_kwargs)
    return model, history, dir_name

def timeit(method):
    def timed(*args, **kw):
        ts = time.perf_counter()
        result = method(*args, **kw)
        te = time.perf_counter()
        diff = te - ts
        print(f"{method.__name__}: {diff:.8f} s")
        return result
    return timed

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)

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

"""
    Save model
"""
@timeit
def save_model(model, model_name, model_type, train_dataset, val_dataset, test_dataset, params, hparams, history, LOG_DIR, loss, metrics):
    target_dir = f"//Users/christianbv/PycharmProjects/EiT/tmp/models/{model_type}/{model_name}"
    os.mkdir(target_dir)

    export_logs_path = os.path.join(target_dir, "export_logs.txt")
    with open(export_logs_path, 'w') as export_logs:
        with redirect_stdout(export_logs):
            # Get number of parameters
            params_counts = {
                "trainable_params": np.sum([K.count_params(w) for w in model.trainable_weights]),
                "non_trainable_params": np.sum([K.count_params(w) for w in model.non_trainable_weights])
            }
            params_counts["total_params"] = params_counts["trainable_params"] + params_counts["non_trainable_params"]

            # Generate baselines
            baseline_model = BaselineModel(params, hparams)
            baseline_model.compile(loss=loss, metrics=metrics)

            # Generate evaluation metrics for validation and test set
            final_metrics_train = model.evaluate(train_dataset, return_dict=True)
            final_metrics_train = {f"final_train_{k}": v for k, v in final_metrics_train.items()}
            final_metrics_val = model.evaluate(val_dataset, return_dict=True)
            final_metrics_val = {f"final_val_{k}": v for k, v in final_metrics_val.items()}
            final_metrics_test = model.evaluate(test_dataset, return_dict=True)
            final_metrics_test = {f"final_test_{k}": v for k, v in final_metrics_test.items()}

            # Generate baseline metrics for validation and test set
            baseline_metrics_train = baseline_model.evaluate(train_dataset, return_dict=True)
            baseline_metrics_train = {f"baseline_train_{k}": v for k, v in baseline_metrics_train.items()}
            baseline_metrics_val = baseline_model.evaluate(val_dataset, return_dict=True)
            baseline_metrics_val = {f"baseline_val_{k}": v for k, v in baseline_metrics_val.items()}
            baseline_metrics_test = baseline_model.evaluate(test_dataset, return_dict=True)
            baseline_metrics_test = {f"baseline_test_{k}": v for k, v in baseline_metrics_test.items()}

            # Generate Dataframe and export to parquet
            logs_params = {
                "num_epochs": len(history.epoch),
                **params,
                **hparams,
                **history.params,
                **params_counts,
                **final_metrics_train,
                **final_metrics_val,
                **final_metrics_test,
                **baseline_metrics_train,
                **baseline_metrics_val,
                **baseline_metrics_test
            }
            logs_df = pd.DataFrame({**history.history, "epoch": history.epoch})
            for param, value in logs_params.items():
                logs_df[param] = value
            logs_df.to_parquet(os.path.join(target_dir, f"{model_name}.parquet"))

            # Dump all parameters and metadata to .json file
            with open(os.path.join(target_dir, 'model_details.json'), 'w') as f:
                json.dump(logs_params, f, cls=NpEncoder, indent=4)

            def _convert_model(model, subdir="model"):
                # Create subdirectory
                subdir_path = os.path.join(target_dir, subdir)
                os.mkdir(subdir_path)

                # Write model summary to file
                model_summary_path = os.path.join(subdir_path, "model_summary.txt")
                with open(model_summary_path, 'w') as model_summary:
                    with redirect_stdout(model_summary):
                        model.summary()

                # Export model summary as image
                model_summary_img_path = os.path.join(subdir_path, "model_summary.png")
                tf.keras.utils.plot_model(model, to_file=model_summary_img_path, show_shapes=True)

                # Generate model paths
                keras_model_path = os.path.join(subdir_path, "keras_model.h5")
                saved_model_path = os.path.join(subdir_path, "saved_model")

                # Save and convert model
                model.save(keras_model_path)
                tf.saved_model.save(model, saved_model_path)

            # Convert full model
            _convert_model(model, subdir="model")

            # Compress TensorBoard logs
            model_log_dir = os.path.join(LOG_DIR, model_name)
            tensorboard_logs_path = os.path.join(target_dir, f"{model_name}.tar.gz")

"""
    Single run of model
"""
def single_run(data_arrays, target_arrays, params):
    # Random serach with parameter lock
    hparams = {
        'recurrent_cell': 'lstm',
        'recurrent_layers': 3,
        'recurrent_units': 64,

        'input_sequence_len': 168,
        'output_sequence_len': 24,

        'batch_size': 32,
        'optimizer': 'sgd',
        'loss': 'mse',
        'learning_rate': 10e-4,
    }

    try:
        # Build datasets
        datasets = build_datasets(data_arrays, target_arrays, params, hparams)
        train_dataset, val_dataset, test_dataset = datasets
        print(hparams)

        fit_kwargs = {
            "x": train_dataset,
            "validation_data": val_dataset,
            "epochs": 300, # 300
            "verbose": 1
        }
        model, history, model_name = train_model(params, hparams, metrics, log_dir=LOG_DIR, model_type=MODEL_TYPE,
                                                 **fit_kwargs)
        save_model(model, model_name, MODEL_TYPE, train_dataset, val_dataset, test_dataset, params, hparams, history,
                      LOG_DIR, hparams['loss'], metrics)
    except Exception as e:
        print(e)


"""
    Training model with mulitple runs
"""
def multiple_runs(data_arrays, target_arrays, params, hparams):
    # Random serach with parameter lock
    hparams_locked = {
        'recurrent_cell': 'lstm',
        # 'recurrent_layers': 3,
        # 'recurrent_units': 64,

        # 'input_sequence_len': 168,
        'output_sequence_len': 48,

        # 'batch_size': 64,
        # 'optimizer': 'adam',
        # 'loss': 'mse',
        # 'learning_rate': 10e-4,
    }

    NUM_ITERATIONS = 300
    for i in range(NUM_ITERATIONS):
        hparams = {k: v.domain.sample_uniform() for k, v in hparams_refs.items() if k not in hparams_locked}
        hparams.update(hparams_locked)

        try:
            # Build datasets
            datasets = build_datasets(data_arrays, target_arrays, params, hparams)
            train_dataset, val_dataset, test_dataset = datasets
            print(hparams)

            fit_kwargs = {
                "x": train_dataset,
                "validation_data": val_dataset,
                "epochs": 300,
                "verbose": 0
            }
            model, history, model_name = train_model(params, hparams, metrics, log_dir=LOG_DIR, model_type=MODEL_TYPE,
                                                     **fit_kwargs)
            save_model(model, model_name, MODEL_TYPE, train_dataset, val_dataset, test_dataset, params, hparams,
                          history, LOG_DIR, hparams['loss'], metrics)
        except Exception as e:
            print(e)


train_df,val_df,test_df = load_data(TARGET_PATH)
data_arrays, target_arrays = create_arrays(train_df, val_df, test_df)
single_run(data_arrays, target_arrays,params)

# Run in terminal: (venv) christianbv@pc-125 EiT % python /Users/christianbv/PycharmProjects/EiT/train_model.py

# Show tensorboard: tensorboard --logdir /Users/christianbv/PycharmProjects/EiT/tmp/logs/rnn_updated