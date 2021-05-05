from sqlalchemy import Boolean, Column, ForeignKey, Integer, String
import sqlalchemy
from sqlalchemy.orm import relationship
from sqlalchemy.sql.sqltypes import Float

from .database import Base

metadata = sqlalchemy.MetaData()

forecast = sqlalchemy.Table(
    "forecasts",
    metadata,
    Column("id", Integer, primary_key=True, index=True),
    Column("timestamp", String),
    Column("nox", Float),
    Column("stationId", ForeignKey('stations.id'), nullable=False)
)

station = sqlalchemy.Table(
    "stations",
    metadata,
    Column("id", Integer, primary_key=True, index=True),
    Column("name", String, unique=True, index=True),
    Column("altitude", Integer),
    Column("longitude", Float),
    Column("latitude", Float),
    Column("locationId", ForeignKey("locations.id")),
)

location = sqlalchemy.Table(
    "locations",
    metadata,
    Column("id", Integer, primary_key=True, index=True),
    Column("name", String, unique=True, index=True)
)