📧 Email Spam Classifier – Machine Learning Project

A complete end-to-end Machine Learning project to classify emails as Spam or Not Spam using TF-IDF and Support Vector Machine (SVM).


🚀 Project Overview

This project detects whether an email is spam or legitimate by applying Natural Language Processing (NLP) and Machine Learning techniques.
It includes:

✔ Training Notebook (spam.ipynb)

✔ Streamlit Web App for predictions

✔ Saved SVM Model (.pkl)

✔ Saved TF-IDF Vectorizer

✔ Full Preprocessing + Evaluation pipeline

This repository is built for learning, portfolio demonstration, and deployment purposes.

🧠 Tech Stack

Python

Scikit-Learn

NLTK

Pandas

NumPy

Streamlit

Matplotlib / Seaborn

📂 Project Files
File	Description
spam.ipynb	Machine Learning model training notebook
app.py	Streamlit app for spam/ham prediction
svm_spam_model.pkl	Trained SVM classification model
tfidf_vectorizer.pkl	Trained TF-IDF vectorizer
requirements.txt	Required dependencies
dataset.csv	Spam/Ham dataset (optional)
⚙️ Installation

Clone the repository:

git clone https://github.com/YOUR-USERNAME/email-spam-classifier-ml.git


Navigate into project folder:

cd email-spam-classifier-ml


Install required packages:

pip install -r requirements.txt

▶️ Running the Streamlit App

Run the following command:

streamlit run app.py


The app will open in your browser and allow you to paste any email text and classify it.

🧪 Model Training

To retrain or re-evaluate the model:

jupyter notebook spam.ipynb


Inside the notebook, you can explore:

Data cleaning

TF-IDF vectorization

SVM model training

Evaluation metrics

Model saving

📊 Model Performance

The SVM classifier delivers:

High accuracy

High precision

High recall

Strong generalization on unseen emails

(Exact scores are included in the notebook.)

🧹 Preprocessing Pipeline

The email text goes through the following steps:

Lowercasing

Removing URLs

Removing punctuation & symbols

Tokenization

Stopword removal

Stemming (Porter Stemmer)

TF-IDF vectorization

🧑‍💻 Author

Shahzaib ASif

🔗 Feel free to connect or explore more of my work.
