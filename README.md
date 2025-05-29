# Text Summarization using Transformers (T5 & BERT)

This project implements and compares transformer models (BERT and T5) for abstractive and extractive text summarization using the WikiHow dataset. The goal is to condense long-form articles into meaningful headlines while evaluating various transformer-based and traditional NLP methods.

---

## 📁 Project Structure

```
text-summarization-project/
├── README.md
├── requirements.txt
├── data/
│   └── wikihow.csv
├── notebooks/
│   ├── summarizer_t5_itr5.ipynb
│   ├── summarizer_trainng_t5_itr6.ipynb
│   ├── summarizer-inference-epoch1and5.ipynb
│   └── summarizer-inference-epoch5vspretrained.ipynb
└── outputs/
    └── trained_models/
```

---

## 📚 Overview

This project investigates two major transformer models:

* **T5 (Text-to-Text Transfer Transformer)**: for **abstractive summarization**.
* **BERT (Bidirectional Encoder Representations from Transformers)**: for **extractive summarization** (tested, but not used due to lower performance).

We also evaluate the performance using metrics like ROUGE and BLEU.

---

## 🧠 Models Used

* **BERT** (tested but found slow and suboptimal for summarization)
* **T5-Small** (fine-tuned on WikiHow data)
* **PEGASUS** (used for comparative evaluation)

---

## 📊 Dataset

**WikiHow Dataset**

* Contains 230,000+ article-summary pairs.
* Average article length: \~580 words
* Average summary length: \~62 words
* We used `headline` as summary and `text` as input.

---

## ⚙️ Methodology

### Preprocessing:

* Dropped unnecessary columns
* Removed nulls
* Tokenized using `T5Tokenizer`

### Training:

* Fine-tuned `T5-small` using PyTorch on GPU (Google Colab/Kaggle)
* Checkpoints saved per epoch

### Inference:

* Performed evaluation on custom articles using trained model
* Compared performance between fine-tuned model and PEGASUS

### Evaluation:

* **ROUGE-1**, **ROUGE-2**, **ROUGE-L**
* **BLEU Score**

---

## 🧪 Results

| Epoch | BLEU Score |
| ----- | ---------- |
| 2     | 0.104      |
| 4     | 0.110      |
| 5     | 0.109      |

The model shows reasonable improvement over epochs, with best performance seen in Epoch 4.

---

## 🆚 Comparative Study

We compared our model's performance against multiple published methods and PEGASUS. The results show competitive performance with state-of-the-art models.

| Model             | ROUGE-1 | ROUGE-2 | ROUGE-L | BLEU  |
| ----------------- | ------- | ------- | ------- | ----- |
| T5 (ours)         | 31.34   | 11.72   | 26.08   | 10.96 |
| Sequence-to-Seq   | 35.46   | 13.30   | 32.65   | -     |
| Pointer Generator | 43.71   | 26.40   | 39.38   | -     |
| PEGASUS (XSum)    | 45.41   | 19.18   | -       | -     |

---

## 🚀 How to Run

### ✅ On Google Colab

```python
# Step 1: Clone the repository
!git clone https://github.com/your-username/text-summarization-project.git

# Step 2: Navigate into directory
%cd text-summarization-project

# Step 3: Open notebooks and run cells
```

### ✅ On Kaggle Kernels

* Upload the repository or import via GitHub
* Ensure GPU is turned on in Notebook Settings
* Run notebooks from `notebooks/`

---

## 📦 Requirements

Add this to your `requirements.txt`:

```txt
torch
transformers
sklearn
pandas
numpy
rouge-score
nltk
```

---

## 👨‍💻 Authors

* **Jhilmit Asri** – Training code, fine-tuning, documentation, presentation
* **Supriya Ganga** – Evaluation, presentation, support on code
* **Asrith Reddy Velireddy** – Literature review, evaluation, visual analysis

---

## 🔗 Resources & References

* [T5 Paper](https://arxiv.org/abs/1910.10683)
* [PEGASUS Paper](https://arxiv.org/abs/1912.08777)
* [ROUGE Score](https://aclanthology.org/W04-1013/)
* [BLEU Score](https://dl.acm.org/doi/10.3115/1073083.1073135)

---

## 📥 License

This project is for academic and educational purposes.
