from fastapi import FastAPI
from pydantic import BaseModel
from routes.dataset import data_router
from generics import response_builder
app = FastAPI()
app.include_router(data_router)

class Example(BaseModel):
    instruction: str
    response: str

class IngestExamples(BaseModel):
    raw: str | None = None
    example: list[Example] | None = None
    dataset_id: int
    dataset_description: str
    dataset_name: str
@app.get('/')
def get_root(): 
    return {"message": "Data Processing API"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)