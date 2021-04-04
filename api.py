from fastapi import FastAPI, Depends
from sqlalchemy.orm import Session

from sql import crud, models, schemas
from sql.database import SessionLocal, engine

models.Base.metadata.create_all(bind=engine)

api = FastAPI()

## TODO: ASYNC DB-IMPLEMENTATION
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@api.get("/")
async def root():
    return {"message": "Hello World"}


@api.get("/forecast/{%id}")
async def root():
    return {"message": "Hello World"}

@api.get("/forecast/latest", response_model=schemas.Forecast)
async def get_latest_forecast(db: Session = Depends(get_db)):
    return crud.get_latest_forecast(db)

@api.post("/forecast/", response_model=schemas.Forecast)
async def add_forecast(forecast: schemas.ForecastCreate, db: Session = Depends(get_db)):
    return crud.add_forecast(db, forecast)


@api.get("/stations/")
async def root():
    return {"message": "List of stations"}
