The task data was first preprocessed by cleaning the CSV and converting task descriptions into numeric representations using SentenceTransformers embeddings. Task skills and categories were one-hot or multi-hot encoded and combined with embeddings to form a task feature vector. A members dataset was simulated, including skills and task statistics, and member features were numerically encoded. Next, a task-member pair dataset was created by combining each task with every member, assigning labels (1 if assigned, 0 otherwise) for training purposes. A feedforward neural network (MLP) was then built to take a task-member pair as input and predict the probability that a member is suitable for the task. The model was trained on the simulated dataset, saved, and tested with a new task, successfully producing the top-N suggested members. This forms a complete AI-based task allocation pipeline.

          ┌─────────────────────────────────────┐
          │   Raw Tasks                         │
          │ CSV (description, skills, category) │
          └─────────────────┬───────────────────┘
                            │
                            ▼
                ┌─────────────────────┐
                │ Preprocess Tasks    │
                │ - Clean data        │
                │ - Task embeddings   │
                │ - Skills & category │
                └─────────────────────┘
                            │
                            ▼
                ┌────────────────────┐
                │ Simulate Members   │
                │ - Skills           │
                │ - Tasks done       │
                │ - Tasks assigned   │
                └───────┬────────────┘
                        │
                        ▼
                ┌────────────────────┐
                │ Create Task-Member │
                │   Pairs Dataset    │
                │ - Combine task &   │
                │   member features  │
                │ - Assign labels    │
                └───────┬────────────┘
                        │
                        ▼
            ┌─────────────────────────────┐
            │ Train Neural Net            │
            │ - Input: task-member vector │
            │ - Output: probability       │
            └───────────┬─────────────────┘
                        │
                        ▼
            ┌───────────────────────────────────┐
            │ Predict Suggestions               │
            │ - Input new task                  │
            │ - Compute probability for members │
            │ - Suggest top-N members           │    
            └───────────────────────────────────┘
