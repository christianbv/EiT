from sqlalchemy.orm import Session, query
from sqlalchemy.sql.expression import desc

from . import models, schemas


# STATION METHODS
def get_station_by_id(db: Session, station_id: int):
    return db.query(models.Station).filter(models.Station.id == station_id).first()

def get_stations(db: Session, skip: int = 0, limit: int = 100):
    return db.query(models.Station).offset(skip).limit(limit).all()

def add_station(db: Session, station: schemas.StationCreate):
    db_station = models.Station(
        name=station.name,
        altitude=station.altitude,
        longitude=station.longitude,
        latitude=station.latitude
    )
    db.add(db_station)
    db.commit()
    db.refresh(db_station)
    return db_station

# FORECAST METHODS
def get_latest_forecast(db: Session):
    return db.query(models.Forecast).order_by(models.Forecast.id.desc()).first()

def add_forecast(db: Session, forecast: schemas.ForecastCreate, station_id: int = 1):
    db_forecast = models.Forecast(
        data_one=forecast.data_one, 
        data_two=forecast.data_two, 
        data_three=forecast.data_three, 
        data_four=forecast.data_four, 
        data_five=forecast.data_five,
        stationId=station_id
        )
    db.add(db_forecast)
    db.commit()
    db.refresh(db_forecast)
    return db_forecast