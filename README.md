# FakeNews

A Python-based fake news detection project using TF-IDF feature extraction and a Passive Aggressive Classifier. The system includes a training script and a Streamlit web app for interactive classification.

## Project structure

- `app.py` — Streamlit web application
- `train_model.py` — Train the model and save artifacts
- `preprocess.py` — Text cleaning utility functions
- `data/fake_news_dataset.csv` — Public fake news dataset used for training
- `archive/` — Additional model training datasets used in `train_model.py`
- `model/` — Serialized `vectorizer.pkl` and `model.pkl` artifacts
- `requirements.txt` — Python package dependencies

## Setup

1. Install dependencies:

```bash
python3 -m pip install -r requirements.txt
```

2. Train the model:

```bash
python3 train_model.py
```

3. Run the Streamlit app:

```bash
streamlit run app.py
```

## New features

- Text-based fake news classification using TF-IDF and Passive Aggressive Classifier.
- Image analysis support for uploaded news images.
- Combined text + image risk summary to help evaluate suspicious content.
- Training dataset summary and sample examples displayed in the sidebar.

## Dataset

The dataset is a public fake news dataset containing two columns: `label` and `news`. The label column is binary (`True` for real articles, `False` for fake articles).

## Notes

- The preprocessing pipeline lowercases text, removes URLs, HTML tags, punctuation, and English stopwords.
- The app uses a sigmoid transformation on the model's decision function to generate confidence scores.
- If the NLTK stopwords corpus is not installed, the scripts will download it automatically.
