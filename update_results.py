"""Tum 5 modelin sonuclarini birlestir."""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import pandas as pd
import numpy as np
import joblib
import scipy.sparse
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

y = np.load('features/labels.npy')
idx_test = np.load('features/test_idx.npy')
y_test = y[idx_test]

# Klasik modeller
df = pd.read_csv('data/reviews_preprocessed.csv',
    usecols=['cleaned_text','review_length','word_count','exclamation_count',
             'question_count','avg_word_length','uppercase_ratio',
             'sentiment_polarity','sentiment_subjectivity'])
df = df.dropna(subset=['cleaned_text'])
tfidf = joblib.load('models/tfidf_vectorizer.pkl')
scaler = joblib.load('models/scaler.pkl')
nf = ['review_length','word_count','exclamation_count','question_count',
      'avg_word_length','uppercase_ratio','sentiment_polarity','sentiment_subjectivity']
X = scipy.sparse.hstack([tfidf.transform(df['cleaned_text']), scaler.transform(df[nf])]).tocsr()
Xt = X[idx_test]

lr = joblib.load('models/lr_model.pkl')
svm = joblib.load('models/svm_model.pkl')
lgbm = joblib.load('models/lgbm_model.pkl')

def m(yt, yp):
    return {
        'Accuracy': accuracy_score(yt, yp),
        'Precision': precision_score(yt, yp, average='macro'),
        'Recall': recall_score(yt, yp, average='macro'),
        'F1-Macro': f1_score(yt, yp, average='macro')
    }

models = {
    'Logistic Regression': m(y_test, lr.predict(Xt)),
    'SVM': m(y_test, svm.predict(Xt)),
    'LightGBM': m(y_test, lgbm.predict(Xt)),
}

# DL modeller
try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    Xseq = np.load('features/sequences.npy')[idx_test]

    if os.path.exists('models/lstm_model.keras'):
        lstm = load_model('models/lstm_model.keras')
        models['LSTM'] = m(y_test, np.argmax(lstm.predict(Xseq, verbose=0), axis=1))
        print("LSTM eklendi")

    if os.path.exists('models/bilstm_model.keras'):
        bilstm = load_model('models/bilstm_model.keras')
        models['BiLSTM'] = m(y_test, np.argmax(bilstm.predict(Xseq, verbose=0), axis=1))
        print("BiLSTM eklendi")
except ImportError:
    print("TF yuklenemedi, DL modelleri atlanacak")

comp = pd.DataFrame(models).T.round(4)
comp.to_csv('results/model_comparison.csv')
comp.to_csv('results/final_evaluation.csv')
print(comp.to_string())

best = comp['F1-Macro'].idxmax()
bf1 = comp['F1-Macro'].max()
with open('results/best_model.txt', 'w') as f:
    f.write(f'Best Model: {best}\nF1-Macro: {bf1}')
print(f'\nEn iyi model: {best} (F1={bf1:.4f})')
