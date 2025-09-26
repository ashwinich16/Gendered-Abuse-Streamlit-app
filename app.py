# app.py
import os, json, re, string
import numpy as np
import torch
import torch.nn as nn
import streamlit as st
from transformers import AutoModel, AutoTokenizer
from typing import Dict, List

st.set_page_config(page_title="Gendered Abuse Detection (English)", page_icon="✅")
# ---------- Paths ----------
MODEL_DIR = "fusion_eng_hf"  # folder you exported from the notebook

# ---------- Utils ----------
def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"<handle replaced>", "[USER]", text)
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(rf"[{string.punctuation}]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def texts_to_sequence(text: str, word_index: Dict[str, int], max_len: int, max_features: int) -> np.ndarray:
    """Make a single padded sequence for the BiLSTM/GloVe branch."""
    tokens = text.split()
    idxs = [word_index.get(tok, 0) for tok in tokens]
    # clip to vocab size constraint
    idxs = [i if i < max_features else 0 for i in idxs]
    # pad / truncate
    if len(idxs) < max_len:
        idxs = idxs + [0] * (max_len - len(idxs))
    else:
        idxs = idxs[:max_len]
    return np.array(idxs, dtype=np.int64)

def build_hurtlex_vec(text: str, hurtlex_dict: Dict[str, List[str]], cat2idx: Dict[str, int]) -> np.ndarray:
    vec = np.zeros(len(cat2idx), dtype=np.float32)
    for tok in text.split():
        cats = hurtlex_dict.get(tok)
        if not cats: 
            continue
        for c in cats:
            if c in cat2idx:
                vec[cat2idx[c]] = 1.0
    return vec

# ---------- Model definition (same as your training code) ----------
class BiLSTMEncoder(nn.Module):
    def __init__(self, embedding_matrix: np.ndarray):
        super().__init__()
        vocab_size, embed_size = embedding_matrix.shape
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = False
        self.conv = nn.Conv1d(embed_size, 64, kernel_size=2)
        self.lstm = nn.LSTM(64, 128, batch_first=True, bidirectional=True)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.linear = nn.Linear(256, 128)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.embedding(x)            # (B, T, E)
        x = x.permute(0, 2, 1)           # (B, E, T)
        x = self.conv(x)                 # (B, C, T-1)
        x = x.permute(0, 2, 1)           # (B, T-1, C)
        x, _ = self.lstm(x)              # (B, T-1, 2H)
        x = x.permute(0, 2, 1)           # (B, 2H, T-1)
        x = self.pool(x).squeeze(2)      # (B, 2H)
        x = torch.relu(self.linear(x))   # (B, 128)
        return self.dropout(x)

class IndicBERTEncoder(nn.Module):
    def __init__(self, model_name="ai4bharat/indic-bert"):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.linear = nn.Linear(self.bert.config.hidden_size, 128)
        self.dropout = nn.Dropout(0.2)

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]
        x = torch.relu(self.linear(cls))
        return self.dropout(x)           # (B, 128)

class FusionClassifier(nn.Module):
    def __init__(self, embedding_matrix: np.ndarray, hurtlex_input_dim: int):
        super().__init__()
        self.bert_encoder = IndicBERTEncoder()
        self.bilstm_encoder = BiLSTMEncoder(embedding_matrix)
        self.hurtlex_fc = nn.Sequential(
            nn.Linear(hurtlex_input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 128),
        )
        self.fusion_head = nn.Sequential(
            nn.Linear(640, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, 2),
        )

    def forward(self, input_ids, attention_mask, glove_input, hurtlex_input):
        x1 = self.bert_encoder(input_ids, attention_mask)   # (B, 128)
        x2 = self.bilstm_encoder(glove_input)               # (B, 128)
        x3 = self.hurtlex_fc(hurtlex_input)                 # (B, 128)
        x  = torch.cat([x1, x2, x1 - x2, x1 * x2, x3], dim=1)  # (B, 640)
        return self.fusion_head(x)                          # (B, 2)

# ---------- Load artifacts ----------
@st.cache_resource(show_spinner=True)
def load_artifacts():
    # config
    with open(os.path.join(MODEL_DIR, "config.json"), "r") as f:
        cfg = json.load(f)

    # tokenizer for BERT branch
    hf_tok = AutoTokenizer.from_pretrained(MODEL_DIR)  # uses files in folder

    # embedding matrix for BiLSTM
    emb = np.load(os.path.join(MODEL_DIR, "embedding_matrix.npy"))

    # mappings for HurtLex + word_index for glove branch
    with open(os.path.join(MODEL_DIR, "cat2idx.json"), "r", encoding="utf-8") as f:
        cat2idx = json.load(f)
    with open(os.path.join(MODEL_DIR, "word_index.json"), "r", encoding="utf-8") as f:
        word_index = json.load(f)
    with open(os.path.join(MODEL_DIR, "hurtlex_dict.json"), "r", encoding="utf-8") as f:
        hurtlex_dict = json.load(f)

    # model
    model = FusionClassifier(emb, hurtlex_input_dim=len(cat2idx))
    state_dict = torch.load(os.path.join(MODEL_DIR, "pytorch_model.bin"), map_location="cpu")
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    return cfg, hf_tok, emb, cat2idx, word_index, hurtlex_dict, model

cfg, hf_tok, emb, cat2idx, word_index, hurtlex_dict, model = load_artifacts()

# ---------- Streamlit UI ----------

st.title("Gendered Abuse Detection — English")
st.caption("Custom Fusion: IndicBERT + BiLSTM (GloVe) + HurtLex")
st.caption("example Tweets - R u a female or some other gender?  I don't understand ! Being a woman you choose to be judge of show besides ???  U r not a woman #metoo #Shame on you - Hate")
st.caption("Only an idiot would say are you from Muslim - Gujarat is a place to be from - Muslim is a denotion of the religious following - sorrowful state of affairs for labelling humans as religious entities-funny comparison - Non Hate")
text = st.text_area("Enter text", height=140, placeholder="Type a sentence…")
run = st.button("Analyze")

if run:
    txt = normalize_text(text or "")
    if not txt:
        st.warning("Please enter some text.")
    else:
        # Build inputs
        # 1) BERT
        enc = hf_tok(txt, return_tensors="pt", truncation=True, max_length=cfg.get("max_len", 128))
        # 2) GloVe/BiLSTM
        glove_seq = texts_to_sequence(txt, word_index, max_len=cfg.get("max_len", 128),
                                      max_features=cfg.get("glove_max_features", 4479))
        glove_tensor = torch.tensor(glove_seq).unsqueeze(0)   # (1, T)
        # 3) HurtLex
        hurt_vec = build_hurtlex_vec(txt, hurtlex_dict, cat2idx)
        hurt_tensor = torch.tensor(hurt_vec).unsqueeze(0)     # (1, D)

        with torch.no_grad():
            logits = model(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
                glove_input=glove_tensor,
                hurtlex_input=hurt_tensor,
            )
            probs = torch.softmax(logits, dim=-1).squeeze(0).tolist()

        labels = ["Not Hate", "Hate"]  # order matches your training head (2 outputs)
        top_idx = int(np.argmax(probs))
        st.success(f"Prediction: **{labels[top_idx]}**")

        st.subheader("Scores")
        st.write({labels[i]: float(p) for i, p in enumerate(probs)})

st.markdown("---")
st.caption("Note: Research demo. May be imperfect on slang/code-mix.")
