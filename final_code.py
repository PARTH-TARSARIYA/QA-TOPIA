import numpy as np
import pandas as pd
import torch.nn as nn
import json
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from groq import Groq
import os
import torch
import torch.nn.functional as F
import ast

nltk.download('punkt_tab')
nltk.download('stopwords')

# ── Global variables ──
stop_words = set(stopwords.words('english'))
df = pd.read_csv('Data/final_TM_data.csv')

# ── Fix embedding parsing ──
def parse_embedding(embedding_str):
    if isinstance(embedding_str, str):
        cleaned = embedding_str.strip().strip('[]')
        return np.array([float(x) for x in cleaned.split()], dtype=np.float32)
    return np.array(embedding_str, dtype=np.float32)


df['question_embedding'] = df['question_embedding'].apply(parse_embedding)
df['answer_embedding']   = df['answer_embedding'].apply(parse_embedding)

# ── Model definitions ──
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

# ── Load label maps ──
with open('models/label_maps.json') as f:
    id2label = json.load(f)["id2label"]

# ── Load model ──
prediction_model = TopicMLP()
prediction_model.load_state_dict(
    torch.load('models/best_model.pt', map_location='cpu')['state']
)
prediction_model.eval()

temperature = TempScaler()
temperature.load_state_dict(
    torch.load('models/temperature.pt', map_location='cpu')
)
temperature.eval()

# ── Embedding model ──
embedding_model = SentenceTransformer('sbert_ft')
similarity_model = SentenceTransformer('all-mpnet-base-v2')

# ── Preprocessing ──
def text_preprocess(q, a):
    text = str(q) + ' ' + str(a)
    tokens = word_tokenize(text)
    return [w for w in tokens if w.lower() not in stop_words]

# ── Similarity search ──
def get_similarity(question, answer, topic):

    embed_question = similarity_model.encode(
        [question], normalize_embeddings=True
    )
    embed_answer = similarity_model.encode(
        [answer], normalize_embeddings=True
    )

    new_df = df[df['Generalized_topics'] == topic]

    # Convert embeddings to matrix
    q_embs = np.stack(new_df['question_embedding'].values)
    a_embs = np.stack(new_df['answer_embedding'].values)

    # Vectorized similarity
    sim_q = cosine_similarity(embed_question, q_embs)[0]
    sim_a = cosine_similarity(embed_answer, a_embs)[0]

    # Get top indices
    top_q_idx = np.argsort(sim_q)[::-1][1:11]
    top_a_idx = np.argsort(sim_a)[::-1][1:11]

    # ---------------- REMOVE DUPLICATES ----------------
    seen_questions = set()
    seen_answers = set()

    que_list = []
    for i in top_q_idx:
        if sim_q[i] > 0.5:
            q_text = new_df.iloc[i]['Question']

            if q_text not in seen_questions:
                que_list.append([
                    round(float(sim_q[i] * 100), 4),
                    q_text,
                    new_df.iloc[i]['Topics']
                ])
                seen_questions.add(q_text)

    ans_list = []
    for i in top_a_idx:
        if sim_a[i] > 0.5:
            a_text = new_df.iloc[i]['Answer']

            if a_text not in seen_answers:
                ans_list.append([
                    round(float(sim_a[i] * 100), 4),
                    a_text,
                    new_df.iloc[i]['Topics']
                ])
                seen_answers.add(a_text)

    return que_list, ans_list

# ── Main prediction ──
def predict_topic(question, answer):
    text = ' '.join(text_preprocess(question, answer))

    embedding = embedding_model.encode(
        [text],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype(np.float32)

    x = torch.from_numpy(embedding)

    with torch.no_grad():
        logits = prediction_model(x)
        logits = temperature(logits)
        probs = F.softmax(logits, dim=1)

        confidence, pred = torch.max(probs, dim=1)

    topic = id2label[str(pred.item())]

    return topic, float(confidence.item())