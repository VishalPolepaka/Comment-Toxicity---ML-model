# 🧠 Comment Toxicity Detection 

A machine learning project to classify comments as toxic or non-toxic using a deep learning model. The model is built with TensorFlow/Keras and evaluated based on precision, recall, and accuracy. A Gradio interface is also included for easy testing.

## 📑 Table of Contents

1. [Install Dependencies and Load Data](#install-dependencies-and-load-data)
2. [Preprocess](#preprocess)
3. [Create Sequential Model](#create-sequential-model)
4. [Make Predictions](#make-predictions)
5. [Evaluate Model](#evaluate-model)
6. [Test and Gradio](#test-and-gradio)
7. [Results](#results)

---

## 1. Install Dependencies and Load Data

First, ensure all required Python libraries are installed:

```bash
pip install tensorflow pandas matplotlib scikit-learn
```

Then, load the dataset

---

## 2. Preprocess

Text data is cleaned and tokenized using standard NLP preprocessing steps:

* TextVectorization
* Tokenizing and padding sequences for neural network input

---

## 3. Create Sequential Model

A simple Keras Sequential model is used:

* Embedding Layer
* LSTM
* Dropout
* Bidirectional
* Dense Layers

The model is compiled using:

```python
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
```

---

## 4. Make Predictions

After training, the model is used to predict toxicity on unseen test comments using:

```python
model.predict(vectorized_input)
```

---

## 5. Evaluate Model

Performance is evaluated using:

* **Precision**: 0.8312
* **Recall**: 0.6500
* **Accuracy**: 0.4684

---

## 6. Test and Gradio

A Gradio web UI is included to allow users to test the model in real time by inputting custom comments. Launch it with:

```python
gr.Interface(...).launch()
```
Public url: https://6c2cae68d1439e4e9a.gradio.live/

---

## 📊 Results

| Metric    | Value  |
| --------- | ------ |
| Precision | 0.8312 |
| Recall    | 0.6500 |
| Accuracy  | 0.4684 |

The model demonstrates strong precision, making it suitable for applications where minimizing false positives is critical.

---

## 📌 Note

Model performance can be improved further by:

* Using pre-trained embeddings (e.g., GloVe or BERT)
* Tuning hyperparameters
* Expanding the dataset

---
