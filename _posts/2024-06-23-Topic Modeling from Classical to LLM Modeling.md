---
layout: post
title: "Topic Modeling"
date: 2024-06-23
---

# In Progress
# Topic Modeling 

## LDA
LDA represents documents as a mixture of topics. Similarly, a topic is a mixture of words. If a word has high probability of being in a topic, all the documents having w will be more strongly associated with t as well.

### Caveats

- LDA works off the premise that documents with the same topic will have a lot of words in common.
- LDA is a bag of words model meaning that it only considers individual tokens and not their relationships in a sentence.

## LLMs
Large Language Models (LLMs) like GPT-4, BERT, or other transformer-based models can be effectively utilized for topic modeling by leveraging their ability to understand and generate human-like text. Unlike traditional methods like Latent Dirichlet Allocation (LDA), LLMs can provide more nuanced and context-aware topic representations. Here are some approaches to use LLMs for topic modeling:

### Using Embeddings for Clustering
LLMs can generate embeddings that capture semantic meaning of text. These embeddings can then be clustered to identify topics.

Steps:

1. Text Preprocessing: Clean and preprocess the text data (e.g., removing stopwords, punctuation, and tokenization).
2. Embedding Generation: Use an LLM to generate embeddings for each document or sentence.
3. Dimensionality Reduction: Optionally, use techniques like PCA or t-SNE to reduce the dimensionality of the embeddings.
4. Clustering: Apply clustering algorithms like K-means or DBSCAN on the embeddings to group similar texts.
5. Topic Labeling: Analyze the clusters to identify common themes or keywords representing each topic.

Example with Sentence-BERT:
```Python
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import numpy as np

# Sample documents
documents = [
    "The stock market crashed due to economic instability.",
    "The new movie was a box office hit.",
    "Advancements in AI are accelerating.",
    "Climate change is a pressing issue globally."
]

# Load pre-trained Sentence-BERT model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Generate embeddings
embeddings = model.encode(documents)

# Apply K-means clustering
num_clusters = 2
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
kmeans.fit(embeddings)
labels = kmeans.labels_

# Print clustered documents
for i in range(num_clusters):
    cluster_docs = [documents[j] for j in range(len(documents)) if labels[j] == i]
    print(f"Cluster {i}: {cluster_docs}")
```
### Zero-shot Topic Classification
LLMs like GPT-4 can perform zero-shot classification, where the model assigns texts to predefined topics without needing training data for those topics.

Steps:
Define Topics: List possible topics you want to classify texts into.
Prompt Engineering: Craft prompts to query the LLM for topic classification.
Classification: Use the LLM to classify each document into one of the predefined topics.

Example with GPT-4:
```python
import openai

# OpenAI API key setup (replace with your API key)
openai.api_key = 'your-api-key'

# Sample documents
documents = [
    "The stock market crashed due to economic instability.",
    "The new movie was a box office hit.",
    "Advancements in AI are accelerating.",
    "Climate change is a pressing issue globally."
]

# Define possible topics
topics = ["Finance", "Entertainment", "Technology", "Environment"]

# Function to classify document
def classify_document(document, topics):
    prompt = f"Classify the following text into one of these topics: {', '.join(topics)}.\n\nText: {document}\nTopic:"
    response = openai.Completion.create(engine="davinci", prompt=prompt, max_tokens=1)
    return response.choices[0].text.strip()

# Classify each document
for doc in documents:
    topic = classify_document(doc, topics)
    print(f"Document: {doc}\nClassified Topic: {topic}\n")
```
### Fine-tuning LLMs for Topic Modeling

Fine-tune a pre-trained LLM on a labeled dataset where each document is annotated with a topic. The fine-tuned model can then be used to predict topics for new documents.

Steps:
1. Data Preparation: Create a dataset with documents and their corresponding topic labels.
2. Model Fine-tuning: Fine-tune a pre-trained LLM on the prepared dataset.
3. Prediction: Use the fine-tuned model to classify new documents into topics.

Example using Hugging Face Transformers:
```python
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset, Dataset

# Load dataset (example with a hypothetical dataset)
data = {
    'text': ["The stock market crashed due to economic instability.", "The new movie was a box office hit."],
    'label': [0, 1]  # Assume 0: Finance, 1: Entertainment
}
dataset = Dataset.from_dict(data)

# Preprocess data
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
def preprocess_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True)

encoded_dataset = dataset.map(preprocess_function, batched=True)

# Fine-tune model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
training_args = TrainingArguments(output_dir='./results', num_train_epochs=3, per_device_train_batch_size=2, warmup_steps=500, weight_decay=0.01, logging_dir='./logs')

trainer = Trainer(model=model, args=training_args, train_dataset=encoded_dataset, eval_dataset=encoded_dataset)
trainer.train()

# Predict new documents
new_texts = ["Advancements in AI are accelerating.", "Climate change is a pressing issue globally."]
new_encodings = tokenizer(new_texts, padding='max_length', truncation=True, return_tensors='pt')
predictions = model(**new_encodings)
predicted_labels = predictions.logits.argmax(dim=1).tolist()

print(predicted_labels)
```
## Summary
LLMs can be effectively used for topic modeling through various approaches, including generating embeddings for clustering, zero-shot classification, and fine-tuning for topic prediction. These methods leverage the advanced language understanding capabilities of LLMs to provide more accurate and contextually aware topic modeling.


# Refrences

- [A Beginner’s Guide to Latent Dirichlet Allocation(LDA)](https://medium.com/@corymaklin/latent-dirichlet-allocation-dfcea0b1fddc)
- [Latent Dirichlet Allocation](https://medium.com/@corymaklin/latent-dirichlet-allocation-dfcea0b1fddc)

