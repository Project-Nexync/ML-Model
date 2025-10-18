#!/usr/bin/env python3
"""Assign tasks to persons using embedding+skill-overlap scoring and the Hungarian algorithm.

Usage:
  python src/assign_tasks.py input.json

Input JSON format:
{
  "tasks": [{"id":"t1","title":"...","description":"...","required_skills":[...]}, ...],
  "persons": [{"id":"p1","name":"...","skills":[...],"capacity":2}, ...],
  "alpha": 0.7
}

Output: prints JSON with assignments and unassigned tasks.
"""

import json
import sys
import numpy as np
from scipy.optimize import linear_sum_assignment
from .featurize import Featurizer


def expand_persons_by_capacity(persons):
    slots = []
    slot_owner = []
    for p in persons:
        cap = int(p.get('capacity', 1) or 1)
        for i in range(cap):
            slots.append(p)
            slot_owner.append(p['id'])
    return slots, slot_owner


def assign(tasks, persons, alpha=0.7, min_score=0.0):
    # create featurizer
    f = Featurizer()

    # If tasks don't have required_skills, try to infer from title + available persons
    for t in tasks:
        if not t.get('required_skills'):
            inferred = f.infer_required_skills(
                t.get('title', ''), persons, top_k=3, threshold=0.35)
            if inferred:
                t['required_skills'] = inferred

    # expand persons into slots
    slots, slot_owner = expand_persons_by_capacity(persons)

    # compute score matrix
    scores = f.build_score_matrix(tasks, slots, alpha=alpha)

    # convert to cost for Hungarian
    cost = -scores

    T, S = cost.shape
    n = max(T, S)
    if T != S:
        big = np.zeros((n, n)) + 1e6
        big[:T, :S] = cost
        cost_mat = big
    else:
        cost_mat = cost

    row_ind, col_ind = linear_sum_assignment(cost_mat)

    assignments = []
    assigned_tasks = set()
    for r, c in zip(row_ind, col_ind):
        if r < T and c < S:
            score = scores[r, c]
            if score >= min_score:
                assignments.append({
                    'task_id': tasks[r]['id'],
                    'person_id': slot_owner[c],
                    'score': float(score)
                })
                assigned_tasks.add(tasks[r]['id'])

    unassigned = [t['id'] for t in tasks if t['id'] not in assigned_tasks]

    return {'assignments': assignments, 'unassigned': unassigned}


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python src/assign_tasks.py input.json', file=sys.stderr)
        sys.exit(2)

    path = sys.argv[1]
    with open(path, 'r') as f:
        data = json.load(f)

    tasks = data.get('tasks', [])
    persons = data.get('persons', [])
    alpha = float(data.get('alpha', 0.7))
    min_score = float(data.get('min_score', 0.0))

    result = assign(tasks, persons, alpha=alpha, min_score=min_score)
    print(json.dumps(result, indent=2))
