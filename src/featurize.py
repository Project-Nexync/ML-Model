from sentence_transformers import SentenceTransformer
import numpy as np


def text_for_task(task: dict) -> str:
    """Create a single text string for a task from title, description and required_skills."""
    parts = []
    if task.get("title"):
        parts.append(task.get("title"))
    if task.get("description"):
        parts.append(task.get("description"))
    skills = task.get("required_skills") or task.get("skills") or []
    if isinstance(skills, (list, tuple)):
        parts.append(" ".join([s for s in skills if s]))
    elif isinstance(skills, str) and skills:
        parts.append(skills)
    return " ".join(parts).strip()


def text_for_person(person: dict) -> str:
    """Create a single text string for a person from name and skills."""
    parts = []
    if person.get("name"):
        parts.append(person.get("name"))
    skills = person.get("skills") or []
    if isinstance(skills, (list, tuple)):
        parts.append(" ".join([s for s in skills if s]))
    elif isinstance(skills, str) and skills:
        parts.append(skills)
    return " ".join(parts).strip()


class Featurizer:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def embed_texts(self, texts):
        """Return numpy array of embeddings for a list of texts."""
        if not texts:
            return np.zeros((0, self.model.get_sentence_embedding_dimension()))
        emb = self.model.encode(
            texts, convert_to_numpy=True, show_progress_bar=False)
        return emb

    def compute_cosine_sim(self, A: np.ndarray, B: np.ndarray):
        """Compute cosine similarity between rows of A and rows of B -> shape (A_rows, B_rows)."""
        if A.size == 0 or B.size == 0:
            return np.zeros((A.shape[0], B.shape[0]))
        # normalize
        An = A / np.linalg.norm(A, axis=1, keepdims=True).clip(min=1e-8)
        Bn = B / np.linalg.norm(B, axis=1, keepdims=True).clip(min=1e-8)
        return np.matmul(An, Bn.T)

    @staticmethod
    def skill_overlap(task: dict, person: dict):
        ts = set([s.lower().strip() for s in (
            task.get("required_skills") or task.get("skills") or []) if s])
        ps = set([s.lower().strip()
                 for s in (person.get("skills") or []) if s])
        if not ts:
            return 0.0
        return len(ts & ps) / len(ts)

    def build_score_matrix(self, tasks: list, persons: list, alpha: float = 0.7):
        """Return score matrix (tasks x persons) combining embedding similarity and skill overlap.

        alpha: weight for embedding similarity (0..1). score = alpha * cos_sim + (1-alpha) * overlap
        """
        task_texts = [text_for_task(t) for t in tasks]
        person_texts = [text_for_person(p) for p in persons]

        task_emb = self.embed_texts(task_texts)
        person_emb = self.embed_texts(person_texts)

        cos = self.compute_cosine_sim(task_emb, person_emb)

        # build overlap matrix
        overlap = np.zeros_like(cos)
        for i, t in enumerate(tasks):
            for j, p in enumerate(persons):
                overlap[i, j] = self.skill_overlap(t, p)

        scores = alpha * cos + (1.0 - alpha) * overlap
        return scores

    def infer_required_skills(self, task_title: str, persons: list, top_k: int = 3, threshold: float = 0.35):
        """Infer likely required skills for a task title by comparing the title embedding
        to the set of unique skills present among the provided persons.

        Returns a list of skill strings (may be empty).
        """
        # gather unique skill tokens from persons
        skills_set = set()
        for p in persons:
            for s in (p.get('skills') or []):
                if s:
                    skills_set.add(str(s).strip())
        if not skills_set:
            return []

        skill_list = list(skills_set)

        # embed title and each skill phrase
        title_emb = self.embed_texts([task_title])
        skill_emb = self.embed_texts(skill_list)

        sims = self.compute_cosine_sim(
            title_emb, skill_emb).flatten()  # shape (num_skills,)

        # pick top_k skills above threshold
        indices = sims.argsort()[::-1]
        picked = []
        for idx in indices[:max(top_k, len(indices))]:
            if sims[idx] >= threshold:
                picked.append(skill_list[idx])
        return picked
