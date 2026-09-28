from fastapi import FastAPI
from backend.quality_api import router
from backend.quality_db import init_db

app = FastAPI(title="VisionGuard Model Quality Loop")
 
init_db()
app.include_router(router) 