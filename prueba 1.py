# %% [markdown]
# # Importar Librerias

# %%
import pandas as pd
import numpy as np
import re
import string
import json
import warnings
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, classification_report)

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings("ignore")

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# %%
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Dispositivo de cómputo: {DEVICE}")

# %% [markdown]
# ## Lectura de datos

# %%
file_path = 'movie.csv'
df = pd.read_csv(file_path)

# %% [markdown]
# ## Exploración de datos
# 
# En esta sección se carga el df (`Train_textosODS.xlsx`), se revisan dimensiones, nombres de columnas (`shape`, `columns`) y la estructura de los datos con `info()` (tipos, nulos). Con `head()` se inspeccionan las primeras filas para comprobar el formato de los textos y la variable objetivo ODS.
# 
# ### Hallazgos del analisis exploratorio
# El conjunto de datos está compuesto por textos asociados a 16 Objetivos de Desarrollo Sostenible (ODS), al analizar la distribución de los textos por clase se observa que los ODS 16, 5 y 4 concentran la mayor cantidad de textos, mientras que los ODS 12, 15 y 9 presentan la menor cantidad de textos, No obstante, el desbalance entre clases es moderado y no representa una diferencia extrema entre el ODS con mayor y menor número de textos, adicional el df no contiene textos vacíos ni registros duplicados.
# 
# Se observaron las siguientes características:
# 
# - Promedio de palabras en el texto son 111 palabras
# - El minimo de palabras en un texto fue de 24 palabras
# - El maximo de palabras en un texto fue de 268 palabras
# 
# Finalmente al analizar el corpus se identificó un vocabulario aproximado de 35000 términos únicos, lo que refleja la alta dimensionalidad de característica de los problemas de procesamiento de lenguaje natural.

# %%
# Primeras 5 filas
df.head()

# %%
print("Exploración Inicial")
print(f"Shape             : {df.shape}")
print(f"Columnas          : {df.columns.tolist()}")
print(f"Valores nulos     :\n{df.isnull().sum()}")

# %%
print(f"\nDistribución de clases:\n{df['label'].value_counts()}")

# %%
print(f"\nEjemplo positivo:\n{df[df['label']==1]['text'].iloc[0][:200]}")
print(f"\nEjemplo negativo:\n{df[df['label']==0]['text'].iloc[0][:200]}")

# %%
# Longitud en caracteres de cada texto (número de caracteres por fila)
df["char_count"]   = df["text"].apply(len)
# Longitud en palabras: split por espacios y cuenta de tokens por texto
df["word_count"]   = df["text"].apply(lambda x: len(x.split()))
# COMENTARIO
df["unique_words"] = df["text"].apply(lambda x: len(set(x.lower().split())))
# COMENTARIO
df["avg_word_len"] = df["text"].apply(
    lambda x: np.mean([len(w) for w in x.split()]) if x.split() else 0
)
# COMENTARIO
df["sent_label"] = df["label"].map({1: "Positiva", 0: "Negativa"})

# %%
print("\n--- Estadísticas descriptivas por clase ---")
print(
    df.groupby("sent_label")[["char_count", "word_count", "unique_words", "avg_word_len"]]
    .describe()
    .T
)

# %%
PAL = {"Positiva": "#388697", "Negativa": "#DB162F"}
BG  = "#f8f9fa"

# %%
fig = plt.figure(figsize=(18, 22), facecolor=BG)
gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.52, wspace=0.35)
ax1 = fig.add_subplot(gs[0, 0]) # Create an Axes object using the first grid position from gs
vc = df["sent_label"].value_counts()
bars = ax1.bar(
    vc.index, vc.values,
    color=[PAL[k] for k in vc.index],
    width=0.5, edgecolor="white", linewidth=1.5,
)
for b in bars:
    ax1.text(
        b.get_x() + b.get_width() / 2, b.get_height() + 30,
        f"{int(b.get_height()):,}", ha="center", fontweight="bold",
    )
pct = vc / vc.sum() * 100
ax1.set_title("Distribución de Clases", fontweight="bold", pad=10)
ax1.set_ylabel("# Reseñas"); ax1.set_facecolor(BG)
ax1.spines[["top", "right"]].set_visible(False)
plt.show()

# %%
fig = plt.figure(figsize=(18, 22), facecolor=BG)
gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.52, wspace=0.35)
ax2 = fig.add_subplot(gs[0, 1])
for label, color in PAL.items():
    ax2.hist(
        df[df["sent_label"] == label]["char_count"],
        bins=30, alpha=0.65, color=color, label=label, edgecolor="white",
    )
ax2.set_title("Distribución de Longitud (chars)", fontweight="bold", pad=10)
ax2.set_xlabel("Caracteres"); ax2.set_ylabel("Frecuencia")
ax2.legend(); ax2.set_facecolor(BG)
ax2.spines[["top", "right"]].set_visible(False)
plt.show()

# %%
fig = plt.figure(figsize=(18, 22), facecolor=BG)
gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.52, wspace=0.35)
ax3 = fig.add_subplot(gs[0, 2])
for label, color in PAL.items():
    ax3.hist(
        df[df["sent_label"] == label]["word_count"],
        bins=30, alpha=0.65, color=color, label=label, edgecolor="white",
    )
ax3.set_title("Distribución de N° Palabras", fontweight="bold", pad=10)
ax3.set_xlabel("Palabras"); ax3.set_ylabel("Frecuencia")
ax3.legend(); ax3.set_facecolor(BG)
ax3.spines[["top", "right"]].set_visible(False)
plt.show()

# %%
fig = plt.figure(figsize=(18, 22), facecolor=BG)
gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.52, wspace=0.35)
ax4 = fig.add_subplot(gs[1, 0])
data_box = [df[df["sent_label"] == l]["word_count"].values for l in ["Positiva", "Negativa"]]
bp = ax4.boxplot(
    data_box, labels=["Positiva", "Negativa"], patch_artist=True,
    medianprops=dict(color="white", linewidth=2.5),
)
for patch, color in zip(bp["boxes"], PAL.values()):
    patch.set_facecolor(color); patch.set_alpha(0.75)
ax4.set_title("Box Plot: Palabras por Clase", fontweight="bold", pad=10)
ax4.set_ylabel("N° Palabras"); ax4.set_facecolor(BG)
ax4.spines[["top", "right"]].set_visible(False)
plt.show()

# %%
fig = plt.figure(figsize=(18, 22), facecolor=BG)
gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.52, wspace=0.35)
ax5 = fig.add_subplot(gs[1, 1])
for label, color in PAL.items():
    ax5.hist(
        df[df["sent_label"] == label]["avg_word_len"],
        bins=20, alpha=0.65, color=color, label=label, edgecolor="white",
    )
ax5.set_title("Longitud Promedio de Palabra", fontweight="bold", pad=10)
ax5.set_xlabel("Chars/palabra"); ax5.set_ylabel("Frecuencia")
ax5.legend(); ax5.set_facecolor(BG)
ax5.spines[["top", "right"]].set_visible(False)
plt.show()

# %%
fig = plt.figure(figsize=(18, 22), facecolor=BG)
gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.52, wspace=0.35)
ax6 = fig.add_subplot(gs[1, 2])
for label, color in PAL.items():
    ax6.hist(
        df[df["sent_label"] == label]["unique_words"],
        bins=20, alpha=0.65, color=color, label=label, edgecolor="white",
    )
ax6.set_title("Palabras Únicas por Reseña", fontweight="bold", pad=10)
ax6.set_xlabel("Palabras únicas"); ax6.set_ylabel("Frecuencia")
ax6.legend(); ax6.set_facecolor(BG)
ax6.spines[["top", "right"]].set_visible(False)

# %%
for label, colormap, col in [("Positiva", "Blues", 0), ("Negativa", "Reds", 2)]:
    fig = plt.figure(figsize=(18, 22), facecolor=BG)
    gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.52, wspace=0.35)
    ax = fig.add_subplot(gs[2, col])
    text = " ".join(df[df["sent_label"] == label]["text"])
    wc = WordCloud(
        width=500, height=260, background_color="white",
        colormap=colormap, max_words=80, collocations=False,
    ).generate(text)
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(f"WordCloud – Reseñas {label}s", fontweight="bold",
                 pad=10, color=PAL[label])
    plt.show()

# %% [markdown]
# ## Preparación de los datos
# 
# DESCRIPCION DE LO QUE SE HIZO

# %%
from sklearn.model_selection import train_test_split

train_df, test_df = train_test_split(
    df,
    test_size=0.2,       # 80 % entrenamiento, 20 % prueba
    random_state=SEED,
    stratify=df['label'] # garantiza balance de clases en cada split
)

train_texts = train_df['text'].tolist()
train_labels = train_df['label'].tolist()

test_texts = test_df['text'].tolist()
test_labels = test_df['label'].tolist()

print(f"Train: {len(train_texts)} muestras | Test: {len(test_texts)} muestras")

# %%
import re

def clean_text(text):
    text = text.lower()
    text = re.sub(r"<.*?>", " ", text)        # elimina tags HTML (comunes en IMDb)
    text = re.sub(r"[^a-z\s]", " ", text)     # elimina puntuación
    text = re.sub(r"\s+", " ", text).strip()  # normaliza espacios
    return text

# %%
def build_vocab(texts, max_size=20000):       # vocabulario más grande
    words = []
    for text in texts:
        words.extend(clean_text(text).split())
    freq = Counter(words)
    vocab = {word: i+1 for i, (word, _) in enumerate(freq.most_common(max_size))}
    return vocab
vocab = build_vocab(train_texts)

# %%
def encode(text, vocab, max_len=256):
    tokens = clean_text(text).split()
    idxs = [vocab.get(token, 0) for token in tokens]
    if len(idxs) < max_len:
        idxs += [0] * (max_len - len(idxs))
    else:
        idxs = idxs[:max_len]
    return idxs

# %%
class TextDataset(Dataset):
    def __init__(self, texts, labels, vocab):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        x = torch.tensor(encode(self.texts[idx], self.vocab), dtype=torch.long)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y

train_dataset = TextDataset(train_texts, train_labels, vocab)
test_dataset = TextDataset(test_texts, test_labels, vocab)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64)

# %%
class RNNClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=200, hidden_dim=254, output_dim=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size + 1, embed_dim, padding_idx=0)
        self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.rnn(x)
        out = self.dropout(out[:, -1, :])  # apply dropout on last hidden state
        out = self.fc(out)
        return out

model_rnn = RNNClassifier(vocab_size=len(vocab)).to(DEVICE)
print(model_rnn)

# %%
def plot_training_curves(history, PAL, BG):
    """
    Grafica las curvas de entrenamiento:
    - Loss (train vs val)
    - Accuracy, Precision, Recall y F1 (val)
    """
    epochs_range = range(1, len(history["train_loss"]) + 1)

    fig = plt.figure(figsize=(16, 10), facecolor=BG)
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    # ── 1. Loss ─────────────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(epochs_range, history["train_loss"],
             color=PAL["Negativa"], linewidth=2, marker="o", markersize=4,
             label="Train Loss")
    ax1.plot(epochs_range, history["val_loss"],
             color=PAL["Positiva"], linewidth=2, marker="o", markersize=4,
             label="Val Loss")
    ax1.set_title("Curva de Loss", fontweight="bold", pad=10)
    ax1.set_xlabel("Época"); ax1.set_ylabel("Loss")
    ax1.legend(); ax1.set_facecolor(BG)
    ax1.spines[["top", "right"]].set_visible(False)

    # ── 2. Accuracy ──────────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(epochs_range, history["val_acc"],
             color=PAL["Positiva"], linewidth=2, marker="s", markersize=4,
             label="Val Accuracy")
    # Línea de referencia en el mejor accuracy
    best_acc = max(history["val_acc"])
    best_ep  = history["val_acc"].index(best_acc) + 1
    ax2.axvline(best_ep, color="gray", linestyle="--", linewidth=1, alpha=0.7)
    ax2.axhline(best_acc, color="gray", linestyle="--", linewidth=1, alpha=0.7)
    ax2.annotate(f"Best: {best_acc:.4f}",
                 xy=(best_ep, best_acc),
                 xytext=(best_ep + 0.3, best_acc - 0.02),
                 fontsize=9, color="gray")
    ax2.set_title("Accuracy (Validación)", fontweight="bold", pad=10)
    ax2.set_xlabel("Época"); ax2.set_ylabel("Accuracy")
    ax2.legend(); ax2.set_facecolor(BG)
    ax2.spines[["top", "right"]].set_visible(False)

    # ── 3. Precision y Recall ────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(epochs_range, history["val_precision"],
             color=PAL["Negativa"], linewidth=2, marker="^", markersize=4,
             label="Precision")
    ax3.plot(epochs_range, history["val_recall"],
             color=PAL["Positiva"], linewidth=2, marker="v", markersize=4,
             label="Recall")
    ax3.set_title("Precision y Recall (Validación)", fontweight="bold", pad=10)
    ax3.set_xlabel("Época"); ax3.set_ylabel("Score")
    ax3.legend(); ax3.set_facecolor(BG)
    ax3.spines[["top", "right"]].set_visible(False)

    # ── 4. F1-Score ──────────────────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(epochs_range, history["val_f1"],
             color="#6A0572", linewidth=2, marker="D", markersize=4,
             label="F1-Score")
    best_f1 = max(history["val_f1"])
    best_f1_ep = history["val_f1"].index(best_f1) + 1
    ax4.axvline(best_f1_ep, color="gray", linestyle="--", linewidth=1, alpha=0.7)
    ax4.axhline(best_f1, color="gray", linestyle="--", linewidth=1, alpha=0.7)
    ax4.annotate(f"Best: {best_f1:.4f}",
                 xy=(best_f1_ep, best_f1),
                 xytext=(best_f1_ep + 0.3, best_f1 - 0.02),
                 fontsize=9, color="gray")
    ax4.set_title("F1-Score (Validación)", fontweight="bold", pad=10)
    ax4.set_xlabel("Época"); ax4.set_ylabel("F1")
    ax4.legend(); ax4.set_facecolor(BG)
    ax4.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Curvas de Entrenamiento – LSTM Bidireccional",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.savefig("training_curves.png", dpi=150, bbox_inches="tight",
                facecolor=BG)
    plt.show()
    print("Figura guardada como training_curves.png")

# %%
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ── Funciones de entrenamiento y evaluación ──────────────────────────────────

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for x_batch, y_batch in loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(x_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate_full(model, loader, criterion, device):
    """Retorna loss + las 4 métricas de clasificación."""
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            outputs = model(x_batch)
            total_loss += criterion(outputs, y_batch).item()
            all_preds.extend(outputs.argmax(dim=1).cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())

    avg_loss = total_loss / len(loader)
    acc  = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, zero_division=0)
    rec  = recall_score(all_labels, all_preds, zero_division=0)
    f1   = f1_score(all_labels, all_preds, zero_division=0)
    return avg_loss, acc, prec, rec, f1

# ── Configuración del entrenamiento ─────────────────────────────────────────

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_rnn.to(DEVICE)

optimizer = torch.optim.Adam(model_rnn.parameters(), lr=1e-3, weight_decay=1e-5)
criterion = nn.CrossEntropyLoss()

# Reduce el lr a la mitad si val_loss no mejora en 2 épocas consecutivas
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', patience=2, factor=0.5
)

# ── Inicializar history ──────────────────────────────────────────────────────

history = {
    "train_loss"   : [],
    "val_loss"     : [],
    "val_acc"      : [],
    "val_precision": [],
    "val_recall"   : [],
    "val_f1"       : []
}

# ── Early stopping ───────────────────────────────────────────────────────────

epochs            = 20
patience          = 3
best_val_loss     = float('inf')
epochs_no_improve = 0
best_model_state  = None

# ── Loop de entrenamiento ────────────────────────────────────────────────────

for epoch in range(1, epochs + 1):
    train_loss = train_epoch(model_rnn, train_loader, optimizer, criterion, DEVICE)
    val_loss, acc, prec, rec, f1 = evaluate_full(model_rnn, test_loader, criterion, DEVICE)

    scheduler.step(val_loss)

    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["val_acc"].append(acc)
    history["val_precision"].append(prec)
    history["val_recall"].append(rec)
    history["val_f1"].append(f1)

    print(
        f"Epoch {epoch:02d} | "
        f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
        f"Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | F1: {f1:.4f}"
    )

    # Guarda el mejor modelo y verifica criterio de parada
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_state = model_rnn.state_dict().copy()
        epochs_no_improve = 0
        print(f"           ✓ Mejor modelo guardado (val_loss={best_val_loss:.4f})")
    else:
        epochs_no_improve += 1
        print(f"           Sin mejora {epochs_no_improve}/{patience}")
        if epochs_no_improve >= patience:
            print(f"\nEarly stopping activado en época {epoch}.")
            break

# Restaurar pesos del mejor modelo antes de graficar
model_rnn.load_state_dict(best_model_state)
print("\nMejor modelo restaurado ✓")

# ── Curvas de entrenamiento ──────────────────────────────────────────────────

plot_training_curves(history, PAL, BG)

# %%



