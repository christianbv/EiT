from databases.core import Database
from fastapi import FastAPI, Depends
from sqlalchemy.orm import Session

from sql import models, schemas
from sql.database import SessionLocal, engine
import databases

DATABASE_URL = "sqlite:///./eit.db"

database = databases.Database(DATABASE_URL)

models.metadata.create_all(bind=engine)

api = FastAPI()

@api.on_event("startup")
async def startup():
    await database.connect()


@api.on_event("shutdown")
async def shutdown():
    await database.disconnect()

@api.get("/", response_model=schemas.Forecast)
async def root():
    query = models.forecast.select().order_by(models.forecast.c.id.desc())
    return await database.fetch_one(query)
    #return {"message": "Hello World"}

@api.get("/location/forecast/latest", response_model=schemas.Forecast)
async def get_latest_forecast():
    query = models.forecast.select().order_by(models.forecast.c.id.desc())
    return await database.fetch_one(query)