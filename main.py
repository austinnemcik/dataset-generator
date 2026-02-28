from fastapi import FastAPI
from pydantic import BaseModel
from agent.automation import resume_incomplete_batch_runs
from database import init_db
from routes.dataset import data_router
app = FastAPI()
app.include_router(data_router)


@app.on_event("startup")
async def startup():
    init_db()
    await resume_incomplete_batch_runs()

@app.get('/')
def get_root(): 
    return {"message": "Data Processing API"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
