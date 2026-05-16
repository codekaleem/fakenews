import io
import os
import pickle
import streamlit as st
from preprocess import clean_text, ensure_nltk_resources
from image_utils import analyze_image

MODEL_DIR = 'model'
VECTORIZER_PATH = os.path.join(MODEL_DIR, 'vectorizer.pkl')
MODEL_PATH = os.path.join(MODEL_DIR, 'model.pkl')
DATA_PATH = 'data/fake_news_dataset.csv'
ARCHIVE_DIR = 'archive'


@st.cache_resource
def load_artifacts():
    ensure_nltk_resources()
    if not os.path.exists(VECTORIZER_PATH) or not os.path.exists(MODEL_PATH):
        return None, None
    with open(VECTORIZER_PATH, 'rb') as f:
        vectorizer = pickle.load(f)
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    return vectorizer, model


@st.cache_data
def load_dataset_summary(data_path: str = DATA_PATH, archive_dir: str = ARCHIVE_DIR):
    import pandas as pd
    from pathlib import Path

    paths = []
    primary = Path(data_path)
    if primary.exists():
        paths.append(primary)

    archive = Path(archive_dir)
    if archive.is_dir():
        paths.extend(sorted(archive.glob('*.csv')))

    if not paths:
        return None, None

    frames = []
    for path in paths:
        df = pd.read_csv(path)
        if 'news' not in df.columns:
            if 'text' not in df.columns:
                continue
            text = df['text'].fillna('')
            if 'title' in df.columns:
                text = df['title'].fillna('') + ' ' + text
            df = pd.DataFrame({'news': text, 'label': df['label']})
        frames.append(df[['news', 'label']])

    if not frames:
        return None, None

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(subset=['news'])
    counts = combined['label'].value_counts().to_dict()
    samples = combined[['label', 'news']].sample(n=min(3, len(combined)), random_state=42).to_dict('records')
    return counts, samples


def sigmoid(x):
    import numpy as np
    return 1 / (1 + np.exp(-x))


def predict(text: str, vectorizer, model):
    clean = clean_text(text)
    if not clean:
        return None, None, None

    tokens = clean.split()
    token_count = len(tokens)

    vector = vectorizer.transform([clean])
    prediction = model.predict(vector)[0]
    score = model.decision_function(vector)[0]
    confidence = float(sigmoid(score))
    label = 'Real' if prediction == 1 else 'Fake'
    return label, confidence, clean, token_count


def evaluate_uploaded_image(uploaded_file):
    image_bytes = uploaded_file.getvalue()
    analysis = analyze_image(image_bytes)
    return analysis


def format_byte_size(size: int) -> str:
    for unit in ['bytes', 'KB', 'MB', 'GB']:
        if size < 1024 or unit == 'GB':
            return f'{size:.1f} {unit}'
        size /= 1024


def combine_text_image(text_label, text_confidence, image_score):
    if text_label == 'Fake' or image_score >= 0.8:
        return 'Likely Fake'
    if text_label == 'Real' and image_score <= 0.35:
        return 'Likely Real'
    return 'Mixed / Verify Further'


def main():
    st.set_page_config(page_title='Fake News Detector', layout='wide')
    st.title('Fake News Detector')
    st.write(
        'This app classifies news text as **Real** or **Fake** and analyzes uploaded images for suspicious signals.'
    )

    vectorizer, model = load_artifacts()
    if vectorizer is None or model is None:
        st.error('Model artifacts not found. Run `python train_model.py` first to train and save the model.')
        return

    counts, samples = load_dataset_summary()
    with st.sidebar:
        st.header('Features')
        st.markdown('- Text classification with TF-IDF + Passive Aggressive Classifier')
        st.markdown('- Image authenticity analysis using heuristics')
        st.markdown('- Combined text + image risk summary')
        st.markdown('- Sample training data and class distribution')

        if counts:
            st.subheader('Training data')
            st.write(f"Real: {counts.get(True, 0)}")
            st.write(f"Fake: {counts.get(False, 0)}")
            st.write(f"Total: {sum(counts.values())}")
            with st.expander('Sample training examples'):
                for row in samples:
                    st.write(f"- **{row['label']}**: {row['news'][:120]}...")

    mode = st.radio('Select detection mode', ['Text only', 'Image only', 'Text + Image'])
    st.markdown('---')

    image_analysis = None
    image_file = None
    if mode in ['Image only', 'Text + Image']:
        image_file = st.file_uploader('Upload news-related image', type=['png', 'jpg', 'jpeg'])
        if image_file is not None:
            image_analysis = evaluate_uploaded_image(image_file)
            st.image(image_file, caption='Uploaded image', use_column_width=True)

    text_input = ''
    if mode in ['Text only', 'Text + Image']:
        text_input = st.text_area('Enter news article text here', height=220)

    if st.button('Analyze'):
        if mode == 'Text only':
            if not text_input.strip():
                st.warning('Please provide text to classify.')
            else:
                res = predict(text_input, vectorizer, model)
                if res[0] is None:
                    st.warning('Unable to classify empty or invalid text.')
                else:
                    label, confidence, clean, token_count = res

                    # Heuristics: require a minimum token count and confidence
                    min_tokens = 3
                    min_confidence = 0.60

                    if token_count < min_tokens:
                        st.warning('Input too short — please provide a longer excerpt (≥ 3 words).')
                        with st.expander('Preprocessed text'):
                            st.write(clean)
                    elif confidence < min_confidence:
                        st.warning(f'Low confidence ({confidence*100:.1f}%). Result may be unreliable.')
                        st.write(f'Predicted label: **{label}**')
                        with st.expander('Preprocessed text'):
                            st.write(clean)
                    else:
                        if label == 'Real':
                            st.success(f'Result: **{label}**')
                        else:
                            st.error(f'Result: **{label}**')
                        st.write(f'Confidence: **{confidence * 100:.2f}%**')
                        with st.expander('Preprocessed text'):
                            st.write(clean)

        elif mode == 'Image only':
            if image_analysis is None:
                st.warning('Please upload an image to analyze.')
            else:
                st.subheader('Image analysis results')
                st.write(f"Format: **{image_analysis['format']}**")
                st.write(f"Size: **{format_byte_size(image_analysis['size_bytes'])}**")
                st.write(f"Dimensions: **{image_analysis['width']} x {image_analysis['height']}**")
                st.write(f"Entropy: **{image_analysis['entropy']}**")
                st.write(f"Unique colors: **{image_analysis['unique_colors']}**")
                st.write(f"Color variation: **{image_analysis['stddev']}**")
                st.progress(int(image_analysis['suspicion_score'] * 100))
                if image_analysis['suspicion_score'] >= 0.7:
                    st.error('Suspicion level: High — image appears unusual or synthetic.')
                elif image_analysis['suspicion_score'] >= 0.4:
                    st.warning('Suspicion level: Medium — verify the image source and context.')
                else:
                    st.success('Suspicion level: Low — image appears plausible.')

        else:
            if not text_input.strip() or image_analysis is None:
                st.warning('Please provide both text and an image for combined analysis.')
            else:
                label, confidence, clean = predict(text_input, vectorizer, model)
                if label is None:
                    st.warning('Unable to classify empty or invalid text.')
                else:
                    combined_label = combine_text_image(label, confidence, image_analysis['suspicion_score'])
                    st.subheader('Combined analysis')
                    if combined_label == 'Likely Fake':
                        st.error(f'Overall result: **{combined_label}**')
                    elif combined_label == 'Likely Real':
                        st.success(f'Overall result: **{combined_label}**')
                    else:
                        st.warning(f'Overall result: **{combined_label}**')

                    st.markdown('**Text classification**')
                    st.write(f'- Label: **{label}**')
                    st.write(f'- Confidence: **{confidence * 100:.2f}%**')
                    st.markdown('**Image analysis**')
                    st.write(f'- Suspicion score: **{image_analysis["suspicion_score"] * 100:.0f}%**')
                    st.write(f'- Entropy: **{image_analysis["entropy"]}**')
                    st.write(f'- Unique colors: **{image_analysis["unique_colors"]}**')
                    st.write(f'- Color variation: **{image_analysis["stddev"]}**')
                    with st.expander('Preprocessed text'):
                        st.write(clean)

    st.markdown('---')
    st.markdown('### Notes')
    st.markdown(
        '- Text classification uses a TF-IDF vectorizer and Passive Aggressive Classifier. '
        '- Image analysis uses heuristic signals such as entropy, color variation, and aspect ratio.'
    )
    st.markdown(
        '- Combined mode provides a higher-level risk suggestion, but image-based conclusions are heuristic estimates and should be verified.'
    )


if __name__ == '__main__':
    main()
