## Utils

Helper scripts and small apps used around the dataset and labeling workflows:

- **SCI3000Extractor.py** — Extract figures + captions from SCI-3000 PDFs using JSON annotations; handles resume and caption/figure cropping.
- **SCI3000Grader.py** — Streamlit app to grade figures (accepted / compound / questionable); saves to `grading_labels.json` in the selected folder.
- **SCI3000SingleClassificationReview.py** — Streamlit app to classify single images into the SCI-3000 label set; saves to `single_labels.json` in the folder.
- **imageViewer.py** — Simple Streamlit viewer to page through large image folders when the file manager is impractical.
- **make_splits_relative.py** — Rewrite YOLO split lists (train/val/test) to repo-relative paths for portability.
