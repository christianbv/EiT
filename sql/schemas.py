from typing import List, Optional
from pydantic import BaseModel

class StationBase(BaseModel):
    name: str
    altitude: int
    longitude: float
    latitude: float

class StationCreate(StationBase):
    pass

class Station(StationBase):
    id: int

    class Config: 
        orm_mode = True

class ForecastBase(BaseModel):
    data1: str
    data2: str
    data3: str
    data4: str
    data5: str

class ForecastCreate(BaseModel):
    pass

class Forecast(BaseModel):
    id: int
    stationId: int

    class Config:
        orm_mode = True