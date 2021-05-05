from typing import List, Optional
from pydantic import BaseModel




class Forecast(BaseModel):
    id: int
    timestamp: str
    nox: float
    stationId: int

class Station(BaseModel):
    id: int
    name: str
    altitude: int
    longitude: float
    latitude: float
    locationId: int

class Location(BaseModel):
    id: int
    name: str


class ForecastAdd(BaseModel):
    timestamp: str
    nox: float
    stationId: int

class StationAdd(BaseModel):
    name: str
    altitude: int
    longitude: float
    latitude: float
    locationId: int

class LocationAdd(BaseModel):
    name: str