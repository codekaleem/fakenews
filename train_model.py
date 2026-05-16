import argparse
import os
import pickle

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split

from preprocess import clean_text, ensure_nltk_resources


from pathlib import Path


def collect_dataset_paths(data_path: str, archive_dir: str = 'archive'):
    paths = []
    default_path = Path(data_path)
    if default_path.exists():
        paths.append(default_path)

    archive_path = Path(archive_dir)
    if archive_path.is_dir():
        for csv_file in sorted(archive_path.glob('*.csv')):
            paths.append(csv_file)

    if not paths:
        raise FileNotFoundError(
            f'No dataset files found. Expected {data_path} or CSVs inside {archive_dir}.'
        )
    return paths


def normalize_dataframe(df: pd.DataFrame, source_path: str) -> pd.DataFrame:
    if 'label' not in df.columns:
        raise ValueError(f'Expected label column in {source_path}')

    if 'news' in df.columns:
        normalized = df[['news', 'label']].copy()
    elif 'text' in df.columns:
        text = df['text'].fillna('')
        if 'title' in df.columns:
            text = df['title'].fillna('') + ' ' + text
        normalized = pd.DataFrame({'news': text, 'label': df['label']})
    else:
        raise ValueError(f'Expected news or text column in {source_path}')

    normalized = normalized.dropna(subset=['news'])
    normalized['label'] = normalized['label'].astype(int)
    return normalized


def load_data(data_path: str, archive_dir: str = 'archive') -> pd.DataFrame:
    paths = collect_dataset_paths(data_path, archive_dir)
    data_frames = []
    for path in paths:
        df = pd.read_csv(path)
        data_frames.append(normalize_dataframe(df, str(path)))

    combined = pd.concat(data_frames, ignore_index=True)
    combined = combined.drop_duplicates(subset=['news'])
    print(
        f'Loaded {len(paths)} dataset file(s) with {len(combined)} unique training samples.'
    )
    return combined


def prepare_text(df: pd.DataFrame) -> pd.DataFrame:
    df['clean_text'] = df['news'].apply(clean_text)
    return df


def train(data_path: str, model_dir: str, archive_dir: str = 'archive'):
    ensure_nltk_resources()
    df = load_data(data_path, archive_dir)
    df = prepare_text(df)

    X = df['clean_text']
    y = df['label'].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        sublinear_tf=True,
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = PassiveAggressiveClassifier(max_iter=50, random_state=42)
    model.fit(X_train_vec, y_train)

    y_pred = model.predict(X_test_vec)
    accuracy = accuracy_score(y_test, y_pred)

    print(f'Test accuracy: {accuracy:.4f}')
    print('\nClassification report:')
    print(classification_report(y_test, y_pred, target_names=['Fake', 'Real']))

    os.makedirs(model_dir, exist_ok=True)
    with open(os.path.join(model_dir, 'vectorizer.pkl'), 'wb') as f:
        pickle.dump(vectorizer, f)
    with open(os.path.join(model_dir, 'model.pkl'), 'wb') as f:
        pickle.dump(model, f)

    print(f'Artifacts saved to {model_dir}')


def parse_args():
    parser = argparse.ArgumentParser(description='Train a fake news detection model.')
    parser.add_argument(
        '--data-path',
        default='data/fake_news_dataset.csv',
        help='Path to the primary CSV dataset file.',
    )
    parser.add_argument(
        '--archive-dir',
        default='archive',
        help='Directory containing additional CSV datasets for training.',
    )
    parser.add_argument(
        '--model-dir',
        default='model',
        help='Directory to save model artifacts.',
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    train(args.data_path, args.model_dir)
