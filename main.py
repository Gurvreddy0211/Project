from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from backend.prediction import router as prediction_router
from backend.auth import router as auth_router

app = FastAPI()


app.include_router(prediction_router)
app.include_router(auth_router)

app.mount("/static", StaticFiles(directory="frontend"), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)