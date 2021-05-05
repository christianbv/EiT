import databases
from sql import schemas, models
import sqlalchemy
from syncer import sync
import csv
import codecs
import pandas as pd
from tqdm import tqdm

DATABASE_URL = "sqlite:///./eit.db"

database = databases.Database(DATABASE_URL)



def load_csv_file(path):
    df = pd.read_csv(path,
                     engine = 'python',
                     encoding = 'utf-16',
                     sep = '\t',
                     skiprows= 1,
                     names = ["Date",
                              "No",
                              "No2",
                              "NOx"
                              ])
    df = df.set_index("Date", drop=True)
    for col in df.columns:
        df[col] = df[col].apply(lambda x: str(x).replace(",",".")).astype('float')
    return df

@sync
async def startup():
    await database.connect()

@sync
async def shutdown():
    await database.disconnect()

startup()
# location: name: string
# station: name, altitude, longitude, latitude, locationId
# forecast: timestamp, nox, stationId
@sync
async def insertDummyLocation():
    query = models.location.insert().values(name = 'Danmarksplass')
    await database.execute(query)
    print("Added")

@sync
async def insertDummyStation():
    query = models.station.insert().values(name = 'station_one', altitude = 10, longitude = 123, latitude = 123, locationId = 1)
    await database.execute(query)

@sync
async def insertDummyForecasts():
    data = load_csv_file('./sql/dummy_data.csv')
    #print(len(data.index), len(data['NOx']))
    for i in tqdm(range(11364, len(data.index)), "Adding..."):
        #print(data.index[i], data['NOx'][i])
        query = models.forecast.insert().values(timestamp=data.index[i], nox=data['NOx'][i], stationId=1)
        await database.execute(query)

#insertDummyLocation()
#insertDummyStation()
insertDummyForecasts()
shutdown()