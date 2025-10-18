import json
from src.assign_tasks import assign


def test_assign_basic():
    tasks = [{"id": "t1", "title": "Login page"},
             {"id": "t2", "title": "Email worker"}]
    persons = [{"id": "p1", "name": "A", "skills": ["react"]},
               {"id": "p2", "name": "B", "skills": ["nodemailer"]}]

    result = assign(tasks, persons, alpha=0.5, min_score=0.0)
    assert isinstance(result, dict)
    assert 'assignments' in result and 'unassigned' in result
    # cleaned
