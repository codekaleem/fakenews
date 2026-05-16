# Individual Contributions

This document records the individual contributions for the Fake News Detector project, divided among four team members.

---

**Abdul Hafeez — SP23-BSE-047**
- Role: Project Lead, Dataset Acquisition, Model Design & Evaluation
- Tasks:
  - Led project planning and overall system design.
  - Acquired and verified the WELFake dataset placed in `archive/WELFake_Dataset.csv`.
  - Implemented data-merge logic to combine primary dataset and archived CSVs (`train_model.py`).
  - Trained and evaluated the main text classifier (TF-IDF + `PassiveAggressiveClassifier`) and produced model artifacts in `model/`.
  - Wrote the initial project report and README sections describing methodology and results.

---

**Umer Jalal — SP23-BSE-055**
- Role: Preprocessing & Feature Engineering
- Tasks:
  - Implemented the text preprocessing pipeline in `preprocess.py` (lowercasing, URL/HTML removal, punctuation stripping, and NLTK stopword removal).
  - Configured the TF-IDF extraction parameters (`max_features=5000`, `ngram_range=(1,2)`, `sublinear_tf=True`) in `train_model.py`.
  - Ensured vectorizer training occurs on the training split only and serialized the fitted `vectorizer.pkl`.
  - Assisted in evaluation and produced the classification reports and test-split validation.

---

**Kaleem Ullah — SP23-BSE-051**
- Role: Application & Deployment
- Tasks:
  - Designed and implemented the Streamlit web application in `app.py` to serve the model for interactive use.
  - Integrated model artifact loading (`model/model.pkl`, `model/vectorizer.pkl`) and implemented UI flows for Text-only, Image-only, and Combined analysis modes.
  - Added dataset summary display, sample examples, and helpful UI warnings for short or low-confidence inputs.
  - Validated app imports and runtime behavior and prepared run instructions in `README.md`.

---

**Ali Ahmed — SP23-BSE-031**
- Role: Image Analysis, Testing & Documentation
- Tasks:
  - Implemented an image analysis utility module `image_utils.py` that computes entropy, color statistics, and a heuristic `suspicion_score` for uploaded images.
  - Integrated image analysis into the Streamlit app to show image metadata and suspicion levels and to provide a combined text+image risk summary.
  - Performed functional testing of the end-to-end pipeline and added documentation and updated `requirements.txt`.
  - Helped tune heuristics for short-text handling and low-confidence reporting.

---

If you want these sections inserted into the DOCX project report file (`FakeNewsDetector_Report_AbdulHafeez_SP23BSE047.docx`), I can insert them under the "Individual Contribution" section — tell me where to place them or I can append them as an updated contributions section.
