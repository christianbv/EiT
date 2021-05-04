from typing import List, Optional
from pydantic import BaseModel

class StationBase(BaseModel):
    pass

class StationCreate(StationBase):
    name: str
    altitude: int
    longitude: float
    latitude: float
    locationId: int

class Station(StationBase):
    id: int

    class Config: 
        orm_mode = True

class ForecastBase(BaseModel):
    pass

class ForecastCreate(BaseModel):
    data_one: str
    data_two: str
    data_three: str
    data_four: str
    data_five: str
    stationId: int

class Forecast(BaseModel):
    id: int

    class Config:
        orm_mode = True

class LocationBase(BaseModel):
    pass

class LocationCreate(BaseModel):
    name: str

class Location(BaseModel):
    id: int

    class Config:
        orm_mode = True