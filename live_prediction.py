#import tensorflow as tf
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import glob, os
from datetime import datetime, timedelta
import datetime
import pandas as pd
from gql.transport.requests import RequestsHTTPTransport
from gql import gql, Client
import json
import requests
import pytz




MODEL_TYPE = "rnn_updated"
LOG_DIR = f"../EiT/tmp/logs/{MODEL_TYPE}"
LOG_LEVEL = "ERROR"
TARGET_PATH = "../EiT/final_datasets"

model_24h_path_local = "../EiT/tmp/models/rnn_updated/rnn_updated_20210303_162009/model"

#model_24h = tf.keras.models.load_model(f"{model_24h_path_local}/saved_model")


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

oslo_tz = pytz.timezone('Europe/Oslo')
datetime_stop = datetime.datetime.now().replace(tzinfo=pytz.utc).astimezone(oslo_tz).replace(minute=0, second=0, microsecond=0)

met_elements = "air_temperature,surface_air_pressure,wind_speed,wind_from_direction,relative_humidity,specific_humidity,road_water_film_thickness,sum(duration_of_precipitation PT1H),sum(precipitation_amount PT1H),cloud_area_fraction,surface_snow_thickness,sea_surface_temperature,volume_fraction_of_water_in_soil"

hours = 1
datetime_stop = datetime_stop - timedelta(hours=2)
datetime_start = datetime_stop - timedelta(hours=hours)

datetime_stop = datetime_stop + timedelta(hours=1) # To include last hour (PRA does not include last)
stop = datetime_stop.isoformat()
start = datetime_start.isoformat()

data = get_PRA_data(start, stop, None)
print(data)
