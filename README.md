# 🎬 Movie Review Sentiment Analysis (SVM + Tkinter)

This project is a sentiment analysis tool for IMDB movie reviews using **Support Vector Machines (SVM)** and **TF-IDF vectorization**.  
It includes a simple **Tkinter GUI** where users can enter their own movie review and get a real-time prediction: **Positive 😊** or **Negative 😞**.

---

## 🚀 Features

- Full usage of the IMDB dataset (50,000 reviews)
- Text preprocessing:
  - Lowercasing
  - Removing punctuation, numbers, and extra spaces
  - Stopword removal via TF-IDF
- TF-IDF vectorizer with n-gram support `(1,2)`
- SVM classifier optimized with GridSearchCV
- Evaluation includes:
  - Classification report (accuracy, precision, recall, F1)
  - Confusion matrix heatmap
- User-friendly **Tkinter GUI** with model training and prediction

---

## 🧠 Requirements

- Python 3.8+
- Libraries:
  - pandas  
  - numpy  
  - scikit-learn  
  - matplotlib  
  - seaborn  
  - tkinter (comes built-in with Python)

---

## ▶️ How to Run

1. Place `IMDB Dataset.csv` in the same folder as `emotionAnalysis.py`
2. Run the Python script:

```bash
python emotionAnalysis.py
```

3. Click **"Modeli Eğit"** to train the model on the dataset.
4. Enter a movie review in the input field.
5. Click **"Yorumu Tahmin Et"** to get the sentiment prediction.

---

## 📊 Output

- Confusion matrix visualization
- Accuracy, Precision, Recall, and F1-score printed after training

---

## 📁 Dataset

This project uses the [IMDB Large Movie Review Dataset](https://ai.stanford.edu/~amaas/data/sentiment/),  
containing 50,000 movie reviews labeled as positive or negative.


---

### 👩🏼‍💻 Developer

**Sıla Kebapcı** 

📧 silakebapcii@gmail.com 

💼 [GitHub](https://github.com/silakebapci) 

🔗 [LinkedIn](https://www.linkedin.com/in/silakebapci)

---

