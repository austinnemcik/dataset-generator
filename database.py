from sqlmodel import SQLModel, Field, create_engine, Session, Relationship
from dotenv import load_dotenv
import os

load_dotenv()
_DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_engine(_DATABASE_URL) # what we use to actually query postgres

def get_session():
    with Session(engine) as session:
        yield session

class Dataset(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True) # this will automatically set an id if None
    name: str
    description: str | None = None
    model: str | None = None
    generation_cost: float = 0.0
    grading_cost: float = 0.0
    total_cost: float = 0.0
    examples: list["TrainingExample"] = Relationship(back_populates="dataset", sa_relationship_kwargs={"cascade": "all, delete-orphan"})

class TrainingExample(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    prompt: str
    instruction: str
    response: str
    dataset_id: int = Field(foreign_key="dataset.id")
    dataset: "Dataset" = Relationship(back_populates="examples")
    embedding: str | None = None 

