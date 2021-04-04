from typing import List, Optional
from pydantic import BaseModel

class StationBase(BaseModel):
    pass

class StationCreate(StationBase):
    name: str
    altitude: int
    longitude: float
    latitude: float

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

class Forecast(BaseModel):
    id: int
    stationId: int

    class Config:
        orm_mode = True