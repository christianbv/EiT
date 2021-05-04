import pandas as pd
import numpy as np
import scipy
import matplotlib.pyplot as plt
import datetime
import seaborn as sns
import tensorflow as tf


def load_csv_file(path):
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
    df = df.set_index("Date", drop=True)
    for col in df.columns:
        df[col] = df[col].apply(lambda x: str(x).replace(",",".")).astype('float')
    return df

def read_data():

    train_data_path = "/Users/christianbv/PycharmProjects/EiT/final_datasets/train.parquet"
    df_processed = pd.read_parquet(train_data_path)

    raw_data_path = "/Users/christianbv/PycharmProjects/EiT/raw_datasets/danmarks_plass_v0.csv"
    df_raw = load_csv_file(raw_data_path)
    return df_raw, df_processed

def visualize_wind_vectors(df, is_raw = False):
    df = df.dropna()
    windDirCol = "MET_wind_from_direction_10m"
    windStrengthCol = 'MET_wind_speed_10m'

    if is_raw:
        df[windDirCol] = pd.to_numeric(df[windDirCol], errors='coerce')
        df[windStrengthCol] = pd.to_numeric(df[windStrengthCol], errors='coerce')
        df = df.dropna()
        print("GOOD")
        print(df.isnull().sum(axis = 0))
        print(df.describe().transpose())
        plt.hist2d(df[windDirCol].dropna(), df[windStrengthCol].dropna(), bins=(20, 20),vmax=50)  # vmax is max number of instances (colours)
        plt.colorbar()
        plt.xlabel('Wind Direction [deg]')
        plt.ylabel('Wind Velocity [m/s]')
        plt.title("Wind distribution using degrees and [m/s]")
        plt.savefig("/Users/christianbv/PycharmProjects/EiT/plots/raw_wind_vectors")

        plt.show()
    else:
        # Checking the distribution of the wind
        plt.hist2d(df['windX'].dropna(), df['windY'].dropna(), bins=(20, 20), vmax=100)
        plt.colorbar()
        plt.xlabel('Wind X [m/s]')
        plt.ylabel('Wind Y [m/s]')
        plt.title("Wind distribution after converting to vector-format")
        ax = plt.gca()
        ax.axis('tight')
        plt.savefig("/Users/christianbv/PycharmProjects/EiT/plots/converted_wind_vectors")
        plt.show()

def fourier_transform(data):
    noxCol = 'NEA_NOx'
    #noxCol = 'MET_air_temperature_2m'
    print(data.columns)
    data[noxCol] = data[noxCol].apply(lambda x: str(x).replace(",","."))
    data[noxCol] = data[noxCol].astype('float')

    avgNOx = data[noxCol].interpolate(method='linear')
    print(len(avgNOx))
    fft = tf.signal.rfft(avgNOx)
    print(fft)
    f_per_dataset = np.arange(0, len(fft))

    n_samples_d = len(data[noxCol])
    hours_per_year = 365.2524*24
    years_per_dataset = n_samples_d / (hours_per_year)

    f_per_year = f_per_dataset / years_per_dataset
    print(f_per_year, len(f_per_year))
    plt.step(f_per_year, np.abs(fft))
    plt.xscale('log')
    plt.ylim(0, 1e6)
    plt.xlim([0.1, max(plt.xlim())])
    plt.xticks([1, 365.2524], labels=['1/Year','1/day'])
    plt.xlabel('Frequency (log scale)')
    plt.ylabel('Amplitude')
    plt.savefig("/Users/christianbv/PycharmProjects/EiT/plots/fft_plot")
    plt.show()

def violin_plots(data):

    # Plotting the distribution of each feature:
    data_std = (data - data.mean()) / data.std()
    data_std = data_std.melt(var_name='Column', value_name='Normalized')
    plt.figure(figsize=(24, 12))
    ax = sns.violinplot(x='Column', y='Normalized', data=data_std)
    _ = ax.set_xticklabels(data.keys(), rotation=90)
    plt.savefig("/Users/christianbv/PycharmProjects/EiT/plots/violin_plot")
    plt.show()


df_raw, df_processed = read_data()
#visualize_wind_vectors(df_raw,True)
#visualize_wind_vectors(df_processed,False)
fourier_transform(df_raw)
#violin_plots(df_raw)
