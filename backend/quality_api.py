
from fastapi import APIRouter 
from pydantic import BaseModel
from.quality_db import get_connection

router = APIRouter(prefix="/quality", tags=['Model Quality'])

class Prediction(BaseModel):
  image_path:str
  model_version: str="v1.0"
  predicted_label: str
  confidence: float = 0.0

class Dataset(BaseModel):
      included: bool

@router.post("/predictions")
def add_prediction(data: Prediction):
    db = get_connection()
    cur = db.cursor()

    cur.execute("""
           INSERT INTO predictions
           (image_path,model_version,predicted_label,
    confidence)
           VALUES(?,?,?,?)
      """,(
        data.image_path,
        data.model_version,
        data.predicted_label,
        data.confidence
      ))

    db.commit()
    prediction_id= cur.lastrowid
    db.close()

    return{"id":prediction_id,
    "success":True} 

@router.get("/predictions")
def predictions():
    db = get_connection()
    rows = db.execute(
          "SELECT * FROM predictions ORDER BY id DESC"
        ).fetchall()
    db.close()

    return[dict(row)for row in rows] 

@router.post("/reviews")
def add_review(data:dict):
    db = get_connection()

    db.execute("""
            INSERT INTO reviews
            (prediction_id,review_status,human_label,comment)
            VALUES(?,?,?,?)
         """,(
            data.prediction_id,
            data.review_status,
            data.human_label,
            data.comment
         ))   

    db.commit()
    db.close()

    return {"success":True}
         
@router.get("/reviews")
def reviews():
    db = get_connection()

    rows = db.execute("""
            SELECT
              p.id,
              p.image_path,
              p.model_version,
              p.predicted_label,
              p.confidence,
              r.review_status,
              r.human_label,
              r.comment
              FROM predictions p
              JOIN reviews r 
              ON p.id = r.prediction_id
          """).fetchall()

    db.close()

    return [dict(row) for row in rows]

@router.put("/dataset/{prediction_id}")
def update_dataset(prediction_id: int, data: "Dataset"):
    db = get_connection()

    db.execute("""
               INSERT INTO dataset_items (prediction_id,included)
               VALUES (?,?)
               ON CONFLICT(prediction_id)
               DO UPDATE SET included = excluded.included
           """,(prediction_id,int(data.included)))
    db.commit()
    db.close()

    return{"success":True}
           
from fastapi import FastAPI
from backend.quality_api import router
from backend.quality_db import init_db

app = FastAPI(title="VisionGuard Model Quality Loop")

init_db()

app.include_router(router)