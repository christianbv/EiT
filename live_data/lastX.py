import pandas as pd
from gql.transport.requests import RequestsHTTPTransport
from gql import gql, Client

import requests # In collector layer
import pytz # In collector layer
import json # In Lambda
import argparse # Handled using Lambda event
import dateutil.parser # In Lambda
import os # All file handling must be done towards S3
import datetime # In Lambda
from datetime import timedelta # In Lambda
import numpy as np # In Sagemaker layer
from datetime import timezone # In Lambda


station=""
elements=[]
start=stop=""
CLIENT_ID = 'your-client-id'
oslo_tz = pytz.timezone('Europe/Oslo')
last_x=""

def get_station_ids(column_list):
    nea_stations = []
    met_stations = []
    pra_stations = []

    for col in column_list:
        #print(col)
        if col.startswith('NEA'):
            # This is a NEA station column
            col_elements = col.split(".")
            nea_stations.append(col_elements[1])
        if col.startswith('MET'):
            # This is a MET station column
            col_elements = col.split(".")
            # Remove :x if exists
            met_split = col_elements[1].split(":")
            met_stations.append(met_split[0])
        if col.startswith('PRA'):
            # This is a PRA station column
            col_elements = col.split(".")
            # Remove :x if exists
            pra_split = col_elements[1].split(":")
            pra_stations.append(pra_split[0])
    
    # Remove duplicates
    nea_stations = list(dict.fromkeys(nea_stations))
    met_stations = list(dict.fromkeys(met_stations))
    pra_stations = list(dict.fromkeys(pra_stations))

    return nea_stations, met_stations, pra_stations

def flattenPRA(last_x=last_x):
    #print(f'In flattenPRA...')
    data = json.loads(last_x)

    trafficRegistrationPoint = data['trafficData']['trafficRegistrationPoint']
    edges = data['trafficData']['volume']['byHour']['edges']
    pandas_rows = []
    # Aggreegrate vehicle count for all lanes and directions
    station = trafficRegistrationPoint['id']
    #print(f'Aggregating vehicle count for station {station}')
    for node in edges:
        upTo5_6 = from5_6To7_6 = from7_6To12_5 = from12_5To16 = from16To24 = from24up = 0
        for byDirection in node['node']['byDirection']:
            first56 = True
            for byLengthRange in byDirection['byLengthRange']:
                if (byLengthRange['lengthRange']['lowerBound']) == None:
                    # For some reason not all volumeNumbers exists...
                    if byLengthRange['total']['volumeNumbers']:
                        upTo5_6 = upTo5_6 + byLengthRange['total']['volumeNumbers']['volume']
                elif (byLengthRange['lengthRange']['lowerBound']) == 5.6:
                    # We forgot to query for upperBound but the first one of the 5.6 is
                    # an aggregator for all vehicles larger than 5.6. Skip it.
                    if first56:
                        first56 = False
                    else:
                        # This is the 56to76 volume, sum it in
                        if byLengthRange['total']['volumeNumbers']:
                            from5_6To7_6 = from5_6To7_6 + byLengthRange['total']['volumeNumbers']['volume']
                elif (byLengthRange['lengthRange']['lowerBound']) == 7.6:
                    if byLengthRange['total']['volumeNumbers']:
                        from7_6To12_5 = from7_6To12_5 + byLengthRange['total']['volumeNumbers']['volume']
                elif (byLengthRange['lengthRange']['lowerBound']) == 12.5:
                    if byLengthRange['total']['volumeNumbers']:
                        from12_5To16 = from12_5To16 + byLengthRange['total']['volumeNumbers']['volume']
                elif (byLengthRange['lengthRange']['lowerBound']) == 16:
                    if byLengthRange['total']['volumeNumbers']:
                        from16To24 = from16To24 + byLengthRange['total']['volumeNumbers']['volume']
                elif (byLengthRange['lengthRange']['lowerBound']) == 24:
                    if byLengthRange['total']['volumeNumbers']:
                        from24up = from24up + byLengthRange['total']['volumeNumbers']['volume']

            
        row = {
            "from" : node['node']['from'],
            "id" : trafficRegistrationPoint['id'],
            "name" : trafficRegistrationPoint['name'],
            "lat" : trafficRegistrationPoint['location']['coordinates']['latLon']['lat'],
            "lon" : trafficRegistrationPoint['location']['coordinates']['latLon']['lon'],
            "upTo5_6" : upTo5_6,
            "from5_6To7_6" : from5_6To7_6,
            "from7_6To12_5" : from7_6To12_5,
            "from12_5To16" : from12_5To16,
            "from16To24" : from16To24,
            "from24up" : from24up
        }
        
        pandas_rows.append(row)
    # Generate dataframe using trafficRegistrationPoint and aggregated data
    #print(f'Create dataframe for {station}')
    df = pd.DataFrame(pandas_rows)
    
    # Remove file if it exists (or S3 will just add to it...)
    file_name = f'/tmp/last_x_PRA.{station}.parquet'
    if os.path.exists(file_name):
        #print(f"File: {file_name} exists, REMOVE IT!")
        os.remove(file_name)

    #print(f'Write last_x data for PRA station: {station} to file: {file_name}')
    df.to_parquet(file_name, engine='pyarrow')
    #print(f'Dataframe shape is: {df.shape}')

def get_PRA_data(start, stop, station):
    #print("In get_PRA_data")

    sample_transport = RequestsHTTPTransport(
        url='https://www.vegvesen.no/trafikkdata/api/',
        use_json=True,
        headers={
            "Content-type": "application/json",
        },
        verify=False,
        retries=3
    )

    client = Client(
        transport=sample_transport,
        #    fetch_schema_from_transport=True,
    )

    heading = '''
    trafficData(trafficRegistrationPointId: "%s") {
          trafficRegistrationPoint {
            name
            id
            latestData {
              volumeByHour
            }
            trafficRegistrationType
            manualLabels {
              affectedLanes {
                lane {
                  laneNumber
                }
              }
              validFrom
              validTo
            }
            commissions {
              validFrom
              validTo
              lanes {
                laneNumber
              }
            }
            direction {
              to
              from
            }
            location {
              coordinates {
                latLon {
                  lat
                  lon
                }
              }
            }
          }
    ''' % (station)

    after = ""
    next = True
    response = ""
    first = True

    while next:
        query_string = '''
      {
        %s
          volume {
            byHour(%sfrom: "%s", to: "%s") {
              pageInfo {
                hasNextPage
                endCursor
              }
              edges {
                node {
                  from
                  to
                  byDirection {
                    heading
                    total {
                      coverage {
                        percentage
                        unit
                        unavailable {
                          numerator
                          denominator
                          percentage
                        }
                        uncertain {
                          numerator
                          denominator
                          percentage
                        }
                        included {
                          numerator
                          denominator
                          percentage
                        }
                      }
                    }
                    byLengthRange {
                      lengthRange {
                        lowerBound
                        upperBound
                      }
                      total {
                        volumeNumbers {
                          volume
                          validSpeed {
                            total
                          }
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
      ''' % (heading, after, start, stop)
        #print (f"Query string is: {query_string}")
        query = gql(query_string)

        this_response = client.execute(query)
        #print("this_response is: " + json.dumps(this_response))
        # Let us first of all do some checking on the response to determine if we have data
        try:
          x = this_response['trafficData']['volume']['byHour']['edges']
        except KeyError:
          # No edges means no data, just return empty response
          #print("No edges in this_response")
          return ""
        if not x:
          # edges list exist but list is empty also means no data, just return empty response
          #print("edges exists in this_response but edge list is empty")
          return ""

        #print (f"this_response type is: {type(this_response)}")
        #print("this_response is: " + json.dumps(this_response))
        if first:
          #print("This is the first response...")
          this_response_str = json.dumps(this_response)
          response = response + this_response_str[:this_response_str.rindex("]")] + ","
          first = False
        else:
          #print("This is a subsequent response...")
          this_response_str = json.dumps(this_response["trafficData"]["volume"]["byHour"]["edges"])
          this_response_str = this_response_str[this_response_str.index("[")+1:]
          response = response + this_response_str[:this_response_str.rindex("]")] + ","

        # Check this_response to see if there are more pages and if yes, set after to endCursor
        if this_response["trafficData"]["volume"]["byHour"]["pageInfo"]["hasNextPage"] == True:
            #print("Next is True")
            endCursor = this_response["trafficData"]["volume"]["byHour"]["pageInfo"]["endCursor"]
            after = '''after: "%s", ''' % (endCursor)
            heading = '''trafficData(trafficRegistrationPointId: "%s") {''' % (station)
        else:
            #print("Next is False")
            next = False

    #print("Clean up the response...")
    # remove last comma
    response = response[:response.rindex(",")]
    # Add closing brackets
    response = response + "]}}}}"
    #print("Return the response...")
    return response

def flattenNEA(last_x=last_x):
    df = pd.read_json(last_x, typ='frame', orient='values')

    station = df.iloc[0]['station']
    # Create initial flat frame from first element in df
    flat = pd.json_normalize(data=df.iloc[0]['values'])
    
    # Rename column to reflect the component
    c_name = df.iloc[0]['component']
    flat.rename(columns={'value':c_name}, inplace=True)
    flat = flat.drop(['toTime'], axis=1)
    flat = flat.drop(['qualityControlled'], axis=1)
    flat['fromTime'] =  pd.to_datetime(flat['fromTime'], format='%Y-%m-%dT%H:%M:%S%z')
    flat.set_index('fromTime', inplace=True)

    # Construct the rest of the dataframe (i.e. additional columns for remaining components)
    for index, row in df.iterrows():
        if index == 0:
            # We used the first row already, skip it
            continue
        #print("Index: " + str(index) + " component: " + row['component'])
        c_name = df.iloc[index]['component']
        new_flat = pd.json_normalize(data=df.iloc[index]['values'])
        new_flat = new_flat.drop(['toTime'], axis=1)
        new_flat = new_flat.drop(['qualityControlled'], axis=1)
        new_flat.rename(columns={'value':c_name}, inplace=True)
        new_flat['fromTime'] =  pd.to_datetime(new_flat['fromTime'], format='%Y-%m-%dT%H:%M:%S%z')
        new_flat.set_index('fromTime', inplace=True)
        # Merge on index (time)    
        flat = flat.merge(new_flat, left_index=True, right_index=True, how='outer')
    
    # Remove file if it exists (or S3 will just add to it...)
    file_name = f'/tmp/last_x_NEA.{station}.parquet'
    if os.path.exists(file_name):
        #print(f"File: {file_name} exists, REMOVE IT!")
        os.remove(file_name)

    #print(f'Write monthly data for to file s3://air-quality-norway/{path}/{name}.parquet')
    # Check if some columns are missing and add them (empty) if that is the case so that all parquet files have the same schema
    if 'PM1' not in flat:
        flat["PM1"] = np.nan
    if 'PM2.5' not in flat:
        flat["PM2.5"] = np.nan        
    if 'PM10' not in flat:
        flat["PM10"] = np.nan    
    if 'NO2' not in flat:
        flat["NO2"] = np.nan
    if 'NOx' not in flat:
        flat["NOx"] = np.nan
    if 'NO' not in flat:
        flat["NO"] = np.nan
    
    # Add metadata columns for all rows
    flat["station"] = df.iloc[0]['station']
    flat["eoi"] = df.iloc[0]['eoi']
    flat["lat"] = df.iloc[0]['latitude']
    flat["lon"] = df.iloc[0]['longitude']

    # Rearrange columns so that all parquet files will have the same schema (the json can have different ordering)
    flat = flat[['station', 'eoi', 'lat', 'lon', 'PM1','PM2.5','PM10','NO2','NOx','NO']]

    # Convert index timestamp to pytz (pyarrow only supports pytz timestamps). Use UTC.
    utc = pytz.timezone('UTC')
    flat.index = flat.index.tz_convert(utc)
    # Make sure all sensor value columns have the right type (filling in empty columns made some of the of wrong type)
    for col in ['PM1', 'PM2.5', 'PM10', 'NO2', 'NO', 'NOx']:
        flat[col] = flat[col].astype('double')
    flat.to_parquet(file_name, engine='pyarrow')
    #print(f'Write last_x data for NEA station: {station} to file: {file_name}')
    #print(f'Dataframe shape is: {flat.shape}')
    return station

def getNEA_last_x_hours(station=station, start=start, stop=stop):
    URL = f"https://api.nilu.no/obs/historical/{start}/{stop}/{station}"
    # Make sure the URL is a valid URI (e.g. substitute spaces with %20 etc.)
    URL = requests.utils.requote_uri(URL)
    #print(f"URL is: {URL}")
    try:
        data = requests.get(url = requests.utils.requote_uri(URL), timeout=20, auth=(CLIENT_ID,'')) 
        response = json.loads(data.text)
        #print(json.dumps(response))
    except Exception as e:
        print(str(e))
        sys.exit(-1)
    
    # Check if errors are returned
    try:
        err = response['errors']
    except Exception as e:
        # Key 'errors' does not exist, pass
        pass
    else:
        # Key 'errors' exists, return -1 so Airflow will try to reschedule it
        print(f"API call returned error: {err}")
        
    return data

def format_datetime(referenceTime):
    # 1. convert object type to string, and then to datetime format@
    dt_referenceTime = dateutil.parser.parse(str(referenceTime))
    # 2. localize the datetime to Oslo local time with pytz
    oslo_datetime_obj = dt_referenceTime.replace(tzinfo=pytz.utc).astimezone(oslo_tz)
    return oslo_datetime_obj

def flattenMET(last_x):
    retrieved_result = json.loads(last_x)
    met_observations = retrieved_result['data']
    # flatten the observations part that contains the actual measurement data
    observations_df = pd.json_normalize(met_observations, record_path='observations', meta=['sourceId', 'referenceTime'],
                                        record_prefix='observations.')

    # get the station ID
    stationID = observations_df.sourceId.unique()[0]
    #print(f'Begin to aggregate weather measurements for station {stationID}')

    observations_df = observations_df[
        ['referenceTime', 'sourceId', 'observations.elementId', 'observations.value', 'observations.unit']]

    # convert the format of referenceTime to datetime
    observations_df['referenceTime'] = observations_df.apply(lambda x: format_datetime(x['referenceTime']),axis=1)

    # pivot the observation dataframe
    observations_df.set_index('referenceTime')
    observations_pivot_df = observations_df.pivot(index='referenceTime', columns='observations.elementId',
                                                    values='observations.value')
    # add met_station_id to the pivoted observation
    observations_pivot_df['met_station_id'] = stationID

    # switch order of the column of the pivoted observation dataframe
    observations_list = list(observations_pivot_df.columns)
    observations_list = [observations_list[-1]] + observations_list[:-1]
    observations_pivot_df = observations_pivot_df[observations_list]

    #print(f'Aggregating weather measurements for station {stationID} is completed.')

    # Remove file if it exists
    file_name = f'/tmp/last_x_MET.{stationID}.parquet'
    if os.path.exists(file_name):
        #print(f'File: {file_name} exists, REMOVE IT!')
        os.remove(file_name)
    
    #print(f'Write last_x data for MET station: {stationID} to file: {file_name}')
    observations_pivot_df.to_parquet(file_name, engine='pyarrow')
    #print(f'Last_x_MET dataframe shape is: {observations_pivot_df.shape}')

def getMET_last_x_hours(station=station, elements=elements, start=start, stop=stop):
    URL = f"https://frost.met.no/observations/v0.jsonld?sources={station}&timeresolutions=PT1H&referencetime={start}/{stop}&elements={elements}"
    
    # Make sure the URL is a valid URI (e.g. substitute spaces with %20 etc.)
    URL = requests.utils.requote_uri(URL)
    #print(f"URL is: {URL}")
    try:
        data = requests.get(url = requests.utils.requote_uri(URL), timeout=20, auth=(CLIENT_ID,'')) 
        response = json.loads(data.text)
    except Exception as e:
        print(str(e))
        sys.exit(-1)
    
    # Check if errors are returned
    if 'error' in response.keys():
        # return error so Airflow will try to reschedule it
        print(json.dumps(response))
        sys.exit(-1)
        
    return data

def maximum_absolute_scaling(df, columns):
    # copy the dataframe
    df_scaled = df.copy()
    # apply maximum absolute scaling
    for column in columns:
        df_scaled[column] = df_scaled[column]  / df_scaled[column].abs().max()
    return df_scaled

def min_max_scaling(df, columns):
    # copy the dataframe
    df_norm = df.copy()
    # apply min-max scaling
    for column in columns:
        df_norm[column] = (df_norm[column] - df_norm[column].min()) / (df_norm[column].max() - df_norm[column].min())
        
    return df_norm
  
def z_score(df, columns):
    # copy the dataframe
    df_std = df.copy()
    # apply the z-score method
    for column in columns:
        df_std[column] = (df_std[column] - df_std[column].mean()) / df_std[column].std()
        
    return df_std
  
def robust_scaling(df, columns):
    # copy the dataframe
    df_robust = df.copy()
    # apply robust scaling
    for column in columns:
        df_robust[column] = (df_robust[column] - df_robust[column].median())  / (df_robust[column].quantile(0.75) - df_robust[column].quantile(0.25))
    return df_robust

def transform_Elgeseter(df):
  # Impute
  df['NEA.Elgeseter.PM10'] = df['NEA.Elgeseter.PM10'].interpolate(method='time')
  df['NEA.Elgeseter.PM10'] = df['NEA.Elgeseter.PM10'].fillna(method='bfill')

  for index in range(df.shape[1]):
    # Select column by index position using iloc[]
    columnSeriesObj = df.iloc[: , index]
    # First interpolate
    df[columnSeriesObj.name] = df[columnSeriesObj.name].interpolate(method='time')
    # Then use mean to fill in those that could not be interpolated
    df[columnSeriesObj.name] = df[columnSeriesObj.name].fillna(df[columnSeriesObj.name].mean())

  df = df.fillna(method='bfill')

  # Feature engineering
  # Convert to radians.
  wv = df.pop('MET.SN68860:0.wind_speed')
  wd_rad = df.pop('MET.SN68860:0.wind_from_direction')*np.pi / 180

  # Calculate the wind x and y components.
  df['MET.SN68860:0.Wx'] = wv*np.cos(wd_rad)
  df['MET.SN68860:0.Wy'] = wv*np.sin(wd_rad)

  timestamp_s = df.index.astype(np.int64)

  day = 24*60*60
  year = (365.2425)*day

  df['Day sin'] = np.sin(timestamp_s * (2 * np.pi / day))
  df['Day cos'] = np.cos(timestamp_s * (2 * np.pi / day))
  df['Year sin'] = np.sin(timestamp_s * (2 * np.pi / year))
  df['Year cos'] = np.cos(timestamp_s * (2 * np.pi / year))

  # Normalize
  columns = list(df.filter(regex=("NEA*")).columns)
  columns = [col for col in columns if not col=="NEA.Elgeseter.PM10"]
  df = z_score(df, columns)
  
  # Do the PRA columns
  columns = list(df.filter(regex=("PRA*")).columns)
  df = min_max_scaling(df, columns)

  # Do the MET columns
  columns = list(df.filter(regex=("MET*")).columns)
  df = z_score(df, columns)

  # Fill remaining nan with 0
  df.fillna(0, inplace=True)

  return df

def transform(station, df):
  transform_func = globals()[f'transform_{station}']
  transformed_df = transform_func(df)
  #print(f'transformed_df contains {df.isnull().sum().sum()} nan')
  return transformed_df

