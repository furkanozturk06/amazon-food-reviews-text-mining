"""LSTM ve BiLSTM modellerini egit."""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import time
import json
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SpatialDropout1D, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import accuracy_score, f1_score

print("TF:", tf.__version__)

# Verileri yukle
X_seq = np.load('features/sequences.npy')
idx_train = np.load('features/train_idx.npy')
idx_val = np.load('features/val_idx.npy')
idx_test = np.load('features/test_idx.npy')
y = np.load('features/labels.npy')

X_train_seq = X_seq[idx_train]
X_val_seq = X_seq[idx_val]
X_test_seq = X_seq[idx_test]
y_train = y[idx_train]
y_val = y[idx_val]
y_test = y[idx_test]

# DL icin subset
np.random.seed(42)
tsub = np.random.choice(len(y_train), 25000, replace=False)
vsub = np.random.choice(len(y_val), 5000, replace=False)
X_tr = X_train_seq[tsub]
y_tr = y_train[tsub]
X_vl = X_val_seq[vsub]
y_vl = y_val[vsub]

VOCAB, MAXLEN, NC = 50000, 200, 3
callbacks = [
    EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', patience=2, factor=0.5, verbose=1)
]

# LSTM modeli
print("\n=== LSTM Egitimi ===")
model_lstm = Sequential([
    Embedding(VOCAB, 64, input_length=MAXLEN),
    SpatialDropout1D(0.2),
    LSTM(64, dropout=0.2),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(NC, activation='softmax')
])
model_lstm.compile(optimizer=Adam(0.001),
                   loss='sparse_categorical_crossentropy',
                   metrics=['accuracy'])

t0 = time.time()
h_lstm = model_lstm.fit(X_tr, y_tr, validation_data=(X_vl, y_vl),
                        epochs=10, batch_size=128, callbacks=callbacks, verbose=2)
print(f"LSTM sure: {time.time()-t0:.0f}s")

model_lstm.save('models/lstm_model.keras')
with open('results/history_lstm.json', 'w') as f:
    json.dump({k: [float(v) for v in vals] for k, vals in h_lstm.history.items()}, f)

yp = np.argmax(model_lstm.predict(X_test_seq, verbose=0), axis=1)
acc1 = accuracy_score(y_test, yp)
f1_1 = f1_score(y_test, yp, average='macro')
print(f"LSTM Test Acc: {acc1:.4f}  F1: {f1_1:.4f}")

# BiLSTM modeli
print("\n=== BiLSTM Egitimi ===")
model_bi = Sequential([
    Embedding(VOCAB, 64, input_length=MAXLEN),
    SpatialDropout1D(0.2),
    Bidirectional(LSTM(64, dropout=0.2)),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(NC, activation='softmax')
])
model_bi.compile(optimizer=Adam(0.001),
                 loss='sparse_categorical_crossentropy',
                 metrics=['accuracy'])

t0 = time.time()
h_bi = model_bi.fit(X_tr, y_tr, validation_data=(X_vl, y_vl),
                    epochs=10, batch_size=128, callbacks=callbacks, verbose=2)
print(f"BiLSTM sure: {time.time()-t0:.0f}s")

model_bi.save('models/bilstm_model.keras')
with open('results/history_bilstm.json', 'w') as f:
    json.dump({k: [float(v) for v in vals] for k, vals in h_bi.history.items()}, f)

yp2 = np.argmax(model_bi.predict(X_test_seq, verbose=0), axis=1)
acc2 = accuracy_score(y_test, yp2)
f1_2 = f1_score(y_test, yp2, average='macro')
print(f"BiLSTM Test Acc: {acc2:.4f}  F1: {f1_2:.4f}")

print("\n=== Tamamlandi! ===")
