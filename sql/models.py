from sqlalchemy import Boolean, Column, ForeignKey, Integer, String
from sqlalchemy.orm import relationship
from sqlalchemy.sql.sqltypes import Float

from .database import Base


class Station(Base):
    __tablename__ = "stations"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)
    altitude = Column(Integer)
    longitude = Column(Float)
    latitude = Column(Float)

    forecasts = relationship("Forecast", back_populates="station")


class Forecast(Base):
    __tablename__ = "forecasts"

    id = Column(Integer, primary_key=True, index=True)
    data_one = Column(String)
    data_two = Column(String)
    data_three = Column(String)
    data_four = Column(String)
    data_five = Column(String)

    stationId = Column(Integer, ForeignKey("stations.id"))
    station = relationship("Station", back_populates="forecasts")
