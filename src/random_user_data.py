import pandas as pd
import random

# Load skill vocabulary from tasks
df_tasks = pd.read_csv('../data/processed/tasks_with_id.csv')
skills_vocab = set()
for skills in df_tasks['skills']:
    for s in skills.split(','):
        skills_vocab.add(s.strip())
skills_vocab = list(skills_vocab)

# Generate 20 dummy members
members = []
for i in range(1, 21):
    member_skills = random.sample(skills_vocab, k=random.randint(3, 6))
    members.append({
        "member_id": i,
        "name": f"Member{i}",
        "skills": ", ".join(member_skills),
        "tasks_done": random.randint(0, 30),
        "tasks_assigned": random.randint(0, 5)
    })

# Create dataframe and save
df_members = pd.DataFrame(members)
df_members.to_csv('../data/processed/members_with_id.csv', index=False)
print("✅ members_with_id.csv created with 20 members")
