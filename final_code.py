import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ---------- GLOBALS ----------
df = None
embedding_model = None
similarity_model = None
prediction_model = None
temperature = None
id2label = None

# ---------- MODELS ----------
class ResBlock(nn.Module):
    def __init__(self, i, o, d):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(i, o), nn.BatchNorm1d(o), nn.GELU(),
            nn.Dropout(d), nn.Linear(o, o), nn.BatchNorm1d(o)
        )
        self.skip = nn.Linear(i, o, bias=False) if i != o else nn.Identity()
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.net(x) + self.skip(x))


class TopicMLP(nn.Module):
    def __init__(self):
        super().__init__()
        blks, prev = [], 768
        for h in [512, 256]:
            blks.append(ResBlock(prev, h, 0.30))
            prev = h
        self.bb = nn.Sequential(*blks)
        self.clf = nn.Linear(prev, 9)

    def forward(self, x):
        return self.clf(self.bb(x))


class TempScaler(nn.Module):
    def __init__(self):
        super().__init__()
        self.T = nn.Parameter(torch.tensor(1.0))

    def forward(self, logits):
        return logits / self.T.clamp(min=0.5, max=3.0)

# ---------- LOAD EVERYTHING ----------
def load_all():
    global df, embedding_model, similarity_model
    global prediction_model, temperature, id2label

    print("Loading data...")

    df = pd.read_csv("Data/final_TM_data.csv")

    def parse_embedding(x):
        if isinstance(x, str):
            x = x.strip().strip('[]')
            return np.array([float(i) for i in x.split()], dtype=np.float32)
        return np.array(x, dtype=np.float32)

    df['question_embedding'] = df['question_embedding'].apply(parse_embedding)
    df['answer_embedding'] = df['answer_embedding'].apply(parse_embedding)

    with open("models/label_maps.json") as f:
        id2label = json.load(f)["id2label"]

    prediction_model = TopicMLP()
    prediction_model.load_state_dict(
        torch.load("models/best_model.pt", map_location="cpu")["state"]
    )
    prediction_model.eval()

    temperature = TempScaler()
    temperature.load_state_dict(
        torch.load("models/temperature.pt", map_location="cpu")
    )
    temperature.eval()

    embedding_model = SentenceTransformer("sbert_ft")
    similarity_model = SentenceTransformer("all-mpnet-base-v2")

    print("All loaded!")

# ---------- PREDICT ----------
def predict_topic(question, answer):
    text = question + " " + answer

    embedding = embedding_model.encode(
        [text], normalize_embeddings=True
    ).astype(np.float32)

    x = torch.from_numpy(embedding)

    with torch.no_grad():
        logits = prediction_model(x)
        logits = temperature(logits)
        probs = F.softmax(logits, dim=1)

        confidence, pred = torch.max(probs, dim=1)

    topic = id2label[str(pred.item())]
    return topic, float(confidence.item())

# ---------- SIMILARITY ----------
def get_similarity(question, answer, topic):

    embed_q = similarity_model.encode([question], normalize_embeddings=True)
    embed_a = similarity_model.encode([answer], normalize_embeddings=True)

    new_df = df[df['Generalized_topics'] == topic]

    q_embs = np.stack(new_df['question_embedding'].values)
    a_embs = np.stack(new_df['answer_embedding'].values)

    sim_q = cosine_similarity(embed_q, q_embs)[0]
    sim_a = cosine_similarity(embed_a, a_embs)[0]

    top_q_idx = np.argsort(sim_q)[::-1][1:11]
    top_a_idx = np.argsort(sim_a)[::-1][1:11]

    que_list = []
    ans_list = []

    for i in top_q_idx:
        if sim_q[i] > 0.5:
            row = new_df.iloc[i]
            que_list.append([
                round(float(sim_q[i]*100), 4),
                row["Question"],
                row["Topics"]
            ])

    for i in top_a_idx:
        if sim_a[i] > 0.5:
            row = new_df.iloc[i]
            ans_list.append([
                round(float(sim_a[i]*100), 4),
                row["Answer"],
                row["Topics"]
            ])

    return que_list, ans_list
