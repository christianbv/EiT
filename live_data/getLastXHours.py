import pytz
import argparse
import datetime
from datetime import timedelta
import requests
import sys
import json
import pandas as pd
import pickle

import lastX

hours = 47
station=""
elements=[]
start=stop=""
last_x = ""
oslo_tz = pytz.timezone('Europe/Oslo')

met_elements = "air_temperature,surface_air_pressure,wind_speed,wind_from_direction,relative_humidity,specific_humidity,road_water_film_thickness,sum(duration_of_precipitation PT1H),sum(precipitation_amount PT1H),cloud_area_fraction,surface_snow_thickness,sea_surface_temperature,volume_fraction_of_water_in_soil"

if __name__ =='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--column_list_file', help="File name to file containing column list names for training data",type=str, required=True)
    parser.add_argument('--hours', help="The number of hours to retrieve data for", type=int, required=True)
    args = parser.parse_args()
    hours = args.hours - 1 # The retrieval includes both ends, so we need to reduce by one

    # Read column names into list
    
    with open(args.column_list_file, 'rb') as f:
        column_list = list(pickle.load(f))

    nea_stations, met_stations, pra_stations = lastX.get_station_ids(column_list)
    #print(f'NEA stations: {nea_stations}')
    #print(f'MET stations: {met_stations}')
    #print(f'PRA stations: {pra_stations}')

    datetime_stop = datetime.datetime.now().replace(tzinfo=pytz.utc).astimezone(oslo_tz).replace(minute=0, second=0, microsecond=0)
    #print(f'Orig datatime stop is: {datetime_stop}')
    # Go back two hour in order to make sure we only have complete hours (allowing provider to have a bit of slack)
    datetime_stop = datetime_stop - timedelta(hours=2)
    # And then move back additional number of hours for the start
    datetime_start = datetime_stop - timedelta(hours=hours)

    # Get MET data
    # Format datetime string as used by MET API, Note MET uses UTC so we need to align timezone
    stop = datetime_stop.astimezone(pytz.utc).strftime('%Y-%m-%dT%H')
    start = datetime_start.astimezone(pytz.utc).strftime('%Y-%m-%dT%H')
    for met_station in met_stations:
        try:
            last_x = lastX.getMET_last_x_hours(station=met_station, elements=met_elements, start=start, stop=stop)
        except Exception as e:
            print (f'Could not get MET data: {e}')
            sys.exit(-1)
        # Flatten and store MET data in parquet file
        lastX.flattenMET(last_x.text)

    # Format start, stop as used by NEA API
    stop = datetime_stop.strftime('%Y-%m-%d %H:00')
    start = datetime_start.strftime('%Y-%m-%d %H:00')

    # Get NEA data
    for nea_station in nea_stations:
        try:
            last_x = lastX.getNEA_last_x_hours(station=nea_station, start=start, stop=stop)
        except Exception as e:
            print (f'Could not get NEA data: {e}')
            sys.exit(-1)
        # Flatten and store NEA data in parquet file
        nea_station = lastX.flattenNEA(last_x.text)

    # Format start, stop as used by PRA API
    datetime_stop = datetime_stop + timedelta(hours=1) # To include last hour (PRA does not include last)
    stop = datetime_stop.isoformat()
    start = datetime_start.isoformat()
    # Get PRA data
    for pra_station in pra_stations:
        try:
            last_x = lastX.get_PRA_data(start, stop, pra_station)
        except Exception as e:
            print (f'Could not get PRA data: {e}')
            sys.exit(-1)

        lastX.flattenPRA(last_x)

    # Iterate over the training data column list and get the associated columns from the flattened
    # dataframes (for the provider in question) in the same order as it was in the training data.

    last_x_df  = pd.DataFrame()
    
    for col in column_list:
        col_elements = col.split(".")
        if col_elements[0] == "NEA":
            nea_station = col_elements[1]
        # Read parquet file
        #print(f'Read file: last_x_{col_elements[0]}.{col_elements[1]}.parquet')
        df = pd.read_parquet(f'last_x_{col_elements[0]}.{col_elements[1]}.parquet', engine='pyarrow')
        # Add column to last_x_df
        if col_elements[0] == "PRA":
            #print(f'Get column: {col_elements[0]}.{col_elements[1]}.{col_elements[3]} from {col_elements[3]}')
            df.set_index('from',inplace = True)
            df.index = pd.to_datetime(df.index)
            last_x_df[f'{col_elements[0]}.{col_elements[1]}.{col_elements[3]}'] = df[col_elements[3]]
        else:
            option_split = col_elements[2].split(":")
            #print(f'Add column {col_elements[0]}.{col_elements[1]}.{option_split[0]}')
            if col_elements[2] == "PM2_5":
                #print(f'Get column: {col_elements[0]}.{col_elements[1]}.{option_split[0]} from PM2.5')
                last_x_df[f'{col_elements[0]}.{col_elements[1]}.{option_split[0]}'] = df["PM2.5"]
            else:
                #print(f'Get column: {col_elements[0]}.{col_elements[1]}.{option_split[0]} from {option_split[0]}')
                last_x_df[f'{col_elements[0]}.{col_elements[1]}.{option_split[0]}'] = df[option_split[0]]

    # Save the df as parquet (only needed when testing so that it is possible to inspect it)
    last_x_df.to_parquet(f'lastX.{nea_station}.parquet', engine='pyarrow')

    # Do the same imputation, feature engineering and normalisation that was done on the training data.
    # This should end up with a last x set of values in the exact same format as was used when training.
    # How should this be done in a generic way so that it can be reused across NEA stations?

    #print(f"Going to transform NEA station: {nea_station}")
    # The lastX.transform will use global() to identify the correct function to call (based on the station name)
    last_x_df = lastX.transform(nea_station, last_x_df)
    # Save the df as parquet (only needed when testing so that it is possible to inspect it)
    last_x_df.to_parquet(f'lastX.{nea_station}_transformed.parquet', engine='pyarrow')
    #print(f'last_x_df contains {last_x_df.isnull().sum().sum()} nan')
    # Transform the final dataframe to the format expected by the prediction algorithm
    pred_inputs = '{"inputs\":\"[['
    for elem in last_x_df.values.tolist():
        pred_inputs = pred_inputs + elem.__str__()
    pred_inputs = pred_inputs + "]]\"}"
    f = open(f'{nea_station}_lastX_inputs.txt', "w")
    f.write(pred_inputs)
    f.flush()
    f.close()