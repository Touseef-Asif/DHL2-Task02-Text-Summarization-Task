# DHL2-Task02-Text-Summarization-Task

## Overview
This project focuses on building a text summarization system using the **CNN/Daily Mail dataset**. The goal is to generate concise summaries of lengthy articles using both **extractive** and **abstractive** summarization techniques. The project is implemented in Python using libraries like `spaCy`, `transformers`, and `nltk`.

---

## Task Description
The task involves the following steps:
1. **Preprocessing**: Clean and prepare the dataset for summarization.
2. **Extractive Summarization**: Generate summaries by selecting the most important sentences from the articles.
3. **Abstractive Summarization**: Generate summaries using a pre-trained Pegasus-XSUM model.
4. **Evaluation**: Evaluate the quality of summaries using ROUGE metrics.
5. **Actionable Insights**: Provide recommendations for improving summarization quality.

---

## Dataset
The **CNN/Daily Mail dataset** is used for this task. It contains:
- **Articles**: Full-length news articles.
- **Highlights**: Human-written summaries of the articles.

Dataset Source: [CNN/Daily Mail Dataset](https://www.kaggle.com/datasets/gowrishankarp/newspaper-text-summarization-cnn-dailymail)

---

## Implementation

### 1. Preprocessing
The preprocessing step involves:
- Tokenizing articles into sentences.
- Removing stopwords and short sentences.
- Limiting text length to prevent memory issues.

```python
def preprocess_text(text):
    """Clean text for summarization"""
    sentences = sent_tokenize(str(text)[:100000])  # Limit to 100k characters
    stop_words = set(stopwords.words('english'))
    cleaned = [
        ' '.join([word.lower() for word in sent.split()
                 if word.lower() not in stop_words and len(word) > 2])
        for sent in sentences if len(sent) > 15
    ]
    return ' '.join(cleaned)
