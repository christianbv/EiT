from fastapi import FastAPI

api = FastAPI()


@api.get("/")
async def root():
    return {"message": "Hello World"}


@api.get("/forecast/{%id}")
async def root():
    return {"message": "Hello World"}


@api.get("/stations/")
async def root():
    return {"message": "List of stations"}
