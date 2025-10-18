from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from src.assign_tasks import assign

app = FastAPI(title="Nexync Task-Assignment ML Service")

# we'll lazy-load the Featurizer during startup to avoid import-time failures
featurizer = None


@app.on_event('startup')
def load_featurizer():
    global featurizer
    try:
        from src.featurize import Featurizer
        featurizer = Featurizer()
    except Exception as e:
        # keep featurizer as None; requests will return 503
        featurizer = None
        app.state.featurizer_error = str(e)


class TaskIn(BaseModel):
    id: str
    title: str


class PersonIn(BaseModel):
    id: str
    name: Optional[str]
    skills: List[str]
    capacity: Optional[int] = 1


class AssignRequest(BaseModel):
    tasks: List[TaskIn]
    persons: List[PersonIn]
    alpha: Optional[float] = 0.7
    min_score: Optional[float] = 0.0


class Assignment(BaseModel):
    task_id: str
    person_id: str
    score: float


class AssignResponse(BaseModel):
    assignments: List[Assignment]
    unassigned: List[str]


@app.post('/assign', response_model=AssignResponse)
async def post_assign(req: AssignRequest):
    tasks = [t.dict() for t in req.tasks]
    persons = [p.dict() for p in req.persons]
    try:
        result = assign(tasks, persons, alpha=req.alpha,
                        min_score=req.min_score, featurizer=featurizer)
        return AssignResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
