from sqlalchemy import Boolean, Column, ForeignKey, Integer, String
from sqlalchemy.orm import relationship
from sqlalchemy.sql.sqltypes import Float

from .database import Base

class Station(Base):
    __tablename__ = "stations"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String)
    altitude = Column(Integer)
    longitude = Column(Float)
    latitude = Column(Float)

    forecasts = relationship("Forecast", back_populates="station")

class Forecast(Base):
    __tablename__ = "forecasts"
    
    id = Column(Integer, primary_key=True)
    data1 = Column(String)
    data2 = Column(String)
    data3 = Column(String)
    data4 = Column(String)
    data5 = Column(String)

    stationId = Column(Integer, ForeignKey("stations.id"))
    station = relationship("Station", back_populates="forecasts")
