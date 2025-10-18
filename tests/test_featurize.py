import numpy as np
import pytest
from src.featurize import Featurizer, text_for_task, text_for_person


class DummyFeaturizer(Featurizer):
    def __init__(self):
        # do not call super to avoid loading the real model
        self.model_name = "dummy"

    def embed_texts(self, texts):
        # return deterministic simple embeddings based on length
        return np.array([[len(t)] for t in texts], dtype=float)


def test_text_helpers():
    t = {"id": "t1", "title": "Build login UI", "required_skills": ["react"]}
    p = {"id": "p1", "name": "Alice", "skills": ["react", "frontend"]}
    assert "Build login UI" in text_for_task(t)
    assert "Alice" in text_for_person(p)


def test_skill_overlap_and_similarity():
    f = DummyFeaturizer()
    tasks = [{"id": "t1", "title": "Login", "required_skills": ["react"]}]
    persons = [{"id": "p1", "name": "A", "skills": ["react"]},
               {"id": "p2", "name": "B", "skills": ["node"]}]

    # embeddings: task len=5, person lens small -> cosine sim becomes 1D compare
    scores = f.build_score_matrix(tasks, persons, alpha=1.0)
    assert scores.shape == (1, 2)
    # skill overlap should be computed as 1.0 for p1 and 0.0 for p2 when alpha<1
    scores2 = f.build_score_matrix(tasks, persons, alpha=0.0)
    assert scores2[0, 0] == pytest.approx(1.0)
    assert scores2[0, 1] == pytest.approx(0.0)


def test_infer_required_skills():
    f = DummyFeaturizer()
    persons = [{"id": "p1", "skills": ["react", "frontend"]},
               {"id": "p2", "skills": ["node"]}]
    inferred = f.infer_required_skills(
        "Build login with React", persons, top_k=2, threshold=0.0)
    # with dummy embedding by length, function should return some skills (threshold 0)
    assert isinstance(inferred, list)
    # Removed accidental footer
