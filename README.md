<h1 align="center">🔤 Handwritten Text Recognition with CRNN + CTC</h1>

<p align="center">
  A custom OCR model that reads handwritten names from images,<br/>
  combining a CNN feature extractor, a bidirectional LSTM and CTC loss.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/TensorFlow-FF6F00?style=flat-square&logo=tensorflow&logoColor=white" />
  <img src="https://img.shields.io/badge/Keras-D00000?style=flat-square&logo=keras&logoColor=white" />
  <img src="https://img.shields.io/badge/OpenCV-5C3EE8?style=flat-square&logo=opencv&logoColor=white" />
  <img src="https://img.shields.io/badge/Character%20Accuracy-88.5%25-2EA043?style=flat-square" />
  <img src="https://img.shields.io/badge/Word%20Accuracy-75.3%25-2EA043?style=flat-square" />
</p>

---

## 📌 Overview

Off-the-shelf OCR handles printed text well but struggles with handwriting. This project builds an OCR model from scratch for **handwritten names**, the kind of field found on forms, cheques and scanned documents.

The model uses the **CRNN** (Convolutional Recurrent Neural Network) architecture trained with **CTC (Connectionist Temporal Classification) loss**. That means it learns to read a whole word without needing character-level bounding boxes, only the image and its text label.

## 📊 Results

Evaluated on 3,000 held-out validation images:

| Metric | Score |
|---|---|
| Correct characters | **88.50%** |
| Correct words (exact match) | **75.27%** |

---

## 🗂️ Dataset

Kaggle **Handwriting Recognition** dataset: scanned images of handwritten first and last names with their transcriptions (`written_name_train_v2.csv`, `written_name_validation_v2.csv`, `written_name_test_v2.csv`).

**Data cleaning:**
- Dropped rows with missing labels.
- Removed images labelled `UNREADABLE`.
- Upper-cased all labels to a fixed alphabet: `A–Z`, `-`, `'` and space.
- Trained on a 30,000-image subset and validated on 3,000 images to keep training time practical.

---

## 🔄 Pipeline

```text
Image (grayscale)
   │  pad/crop to 64×256 on a white canvas, rotate 90°, scale to [0,1]
   ▼
CNN feature extractor
   │  3 × [Conv2D → BatchNorm → ReLU → MaxPool], dropout 0.3
   │  filters: 32 → 64 → 128
   ▼
Reshape to 64 time steps × 1024 features → Dense(64)
   ▼
2 × Bidirectional LSTM (256 units each)
   ▼
Dense + Softmax over 30 classes (29 characters + CTC blank)
   ▼
CTC loss (training)  /  greedy CTC decoding (inference)
```

| Setting | Value |
|---|---|
| Input shape | 256 × 64 × 1 |
| Max label length | 24 characters |
| Output time steps | 64 |
| Optimizer | Adam, learning rate 0.001 |
| Epochs / batch size | 60 / 128 |

The first two RNN outputs are skipped in the CTC loss, because early time steps tend to be noise.

---

## 🧰 Tech Stack

- **TensorFlow / Keras**: model, CTC loss (`ctc_batch_cost`) and decoding (`ctc_decode`)
- **OpenCV**: image loading and preprocessing
- **NumPy / Pandas**: label encoding and dataset handling
- **Matplotlib**: visualizing samples and predictions

---

## 🚀 How to Run

1. Open `main/ocr -- CRNN model.ipynb` in **Kaggle** (recommended, GPU) or Jupyter.
2. Add the Kaggle *Handwriting Recognition* dataset. If you run locally, update the `/kaggle/input/...` paths.
3. Install dependencies if running locally:
   ```bash
   pip install tensorflow opencv-python pandas numpy matplotlib
   ```
4. Run all cells. Training takes about 35 seconds per epoch on a Kaggle GPU.

> The notebook was written for TensorFlow 2.x with standalone Keras imports. On newer versions, replace `Adam(lr=0.001)` with `Adam(learning_rate=0.001)`.

---

## 💡 Key Learnings

- CTC removes the need for character-level alignment, so only image-text pairs are needed.
- Preprocessing (fixed canvas, orientation, normalization) matters as much as architecture.
- Character accuracy is much higher than word accuracy: a single wrong letter fails the whole word.

## 🔭 Possible Improvements

- Train on the full dataset with data augmentation (skew, blur, elastic distortion).
- Beam-search decoding, optionally with a name lexicon or language model.
- Report Character Error Rate (CER) and Word Error Rate (WER) using edit distance.
- Export the model and serve it behind a simple API for document-processing pipelines.

---

## 👤 Author

**Bharat V** · [GitHub](https://github.com/Bharatv05)
