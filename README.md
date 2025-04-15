# 🛑 Hate Speech Detection using Deep Learning
There must be times when you have come across some social media post whose main aim is to spread hate and controversies or use abusive language on social media platforms. As the post consists of textual information to filter out such Hate Speeches NLP comes in handy. This is one of the main applications of NLP which is known as Sentence Classification tasks.

This project aims to detect hate speech in social media text using a deep learning model (LSTM) built with TensorFlow/Keras. The model is trained on labeled tweets to classify whether a given tweet contains hate speech or not.

---
## 📌 Project Highlights

- Preprocessing with NLTK (tokenization, lemmatization, stopword removal)
- Tokenization and padding with Keras Tokenizer
- LSTM-based deep learning model
- Model performance tracking using:
  - EarlyStopping
  - ModelCheckpoint
- Evaluation with accuracy and loss plots

---

## 🧠 Model Architecture

- Embedding Layer
- 2 LSTM Layers with Dropout
- Dense Output Layer with Sigmoid activation (Binary classification)

---

## 🚀 Tech Stack

- Python
- Pandas, NLTK, Scikit-learn
- TensorFlow / Keras
- Google Colab / Jupyter Notebook
- Matplotlib, Seaborn

---
## ⚙️ Installation

```bash
# Clone the repo
git clone https://github.com/Victkhur/Hate-Speech-Detection-Using-Deep-Learning.git
cd Hate-Speech-Detection-Using-Deep-Learning

# Install requirements
pip install -r requirements.txt
