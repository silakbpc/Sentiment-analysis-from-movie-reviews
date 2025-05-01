import pandas as pd
import numpy as np
import re
import string
import tkinter as tk
from tkinter import messagebox
from tkinter import ttk
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

model = None
vectorizer = None

def clean_text(text):
    text = text.lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def train_model():
    global model, vectorizer

    try:
        status_label.config(text="Veri yükleniyor ve model eğitiliyor...", fg="blue")
        window.update()

        df = pd.read_csv("IMDB Dataset.csv")
        df["label"] = df["sentiment"].map({"positive": 1, "negative": 0})
        df["clean_text"] = df["review"].apply(clean_text)

        vectorizer = TfidfVectorizer(stop_words="english", max_features=2000)
        X = vectorizer.fit_transform(df["clean_text"])
        y = df["label"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        params = {"C": [1], "kernel": ["linear"]}  
        grid = GridSearchCV(SVC(), param_grid=params, cv=5, n_jobs=-1)
        grid.fit(X_train, y_train)

        model = grid.best_estimator_
        y_pred = model.predict(X_test)

        report = classification_report(y_test, y_pred, output_dict=False)
        print("Classification Report:\n", report)

        plt.figure(figsize=(6, 4))
        sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues")
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()
        plt.show()

        accuracy = grid.score(X_test, y_test)
        status_label.config(
            text=f"Model eğitildi ✔️\nVeri: {len(df)} yorum\nAccuracy: {accuracy:.2%}", fg="green"
        )
        predict_button.config(state=tk.NORMAL)
        entry.config(state=tk.NORMAL)

    except Exception as e:
        messagebox.showerror("Hata", f"Eğitim sırasında bir hata oluştu:\n{e}")

def predict_sentiment():
    global model, vectorizer
    review = entry.get()
    cleaned = clean_text(review)
    vec = vectorizer.transform([cleaned])
    prediction = model.predict(vec)[0]
    sentiment = "Positive 😊" if prediction == 1 else "Negative 😞"
    messagebox.showinfo("Tahmin", f"Tahmin edilen duygu: {sentiment}")

window = tk.Tk()
window.title("Film Yorumu Duygu Analizi")
window.geometry("500x320")

train_button = tk.Button(window, text="Modeli Eğit ", command=train_model)
train_button.pack(pady=10)

status_label = tk.Label(window, text="Model eğitilmedi.", fg="red", font=("Arial", 10))
status_label.pack()

entry = tk.Entry(window, width=60, state=tk.DISABLED)
entry.pack(pady=10)

predict_button = tk.Button(window, text="Yorumu Tahmin Et", command=predict_sentiment, state=tk.DISABLED)
predict_button.pack(pady=10)

window.mainloop()
