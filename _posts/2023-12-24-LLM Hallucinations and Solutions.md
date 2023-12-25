---
layout: post
title: "LLM Hallucinations: Causes, Mitigation, and Deploymnets"
date: 2023-12-24
---

# Hallucination Causes

LLMs hallucinate when they encounter queries that aren’t part of their training data set — or when their training data set contains erroneous information (this can happen when LLMs are trained on internet data, which, as we all know, can’t always be trusted). LLMs also don’t have memory. Finally, “fine tuning” is often regarded as a way to  reduce hallucinations by retraining a model on new data — but it has its drawbacks.

LLM hallucinations can be caused due to a variety of factors. These include:  

**Incomplete or Noisy Training Data** Lack of complete, relevant, correct, updated, or accurate data in the dataset can lead to gaps or mistakes in the model’s understanding. Consequently, the generated results are erroneous as well.  
**Vague Questions** If the input question or prompt is ambiguous, the model might generate a response based on what it considers the most likely interpretation, which may not align with the user’s intent.  
**Overfitting and Underfitting** Overfitting the training data can make the model too specific, whereas underfitting can make it too general, both of which can lead to hallucinations.  
**Inherent Biases** Models can inherit biases present in the training data, leading them to make assumptions that could result in hallucinations.  
**Absence of Grounding** Unlike humans, these models don’t have real-world experiences or the ability to access real-time data, which limits their understanding and can cause errors.  
**Semantic Gaps** While LLMs are good at pattern recognition, they often lack “common sense” reasoning abilities, which can also contribute to hallucinations.  

In the following section, we’ll look at three methods to stop LLMs from hallucinating: retrieval-augmented generation (RAG), reasoning and iterative querying.  

# Strategies to Mitigate Hallucination

Preventing hallucinations in LLMs like GPT-3 or GPT-4 is an important task. However, it is not an easy one. Here are some strategies that can help:  

**Curated Datasets** Use high-quality, verified datasets for fine-tuning the model. The more accurate the training data, the less likely the model is to generate hallucinations.  
**Output Filtering** Implement mechanisms to filter or flag potentially incorrect or hallucinated outputs based on certain criteria like statistical likelihood or adherence to a domain-specific rule set.  
**Feedback Mechanism** Establish a real-time user feedback system, like re-inforcement learning. If the model produces a hallucination, users can flag the incorrect information, which can be used for further fine-tuning.  
**Iterative Fine-Tuning** Continuously update the model by fine-tuning it with a more recent and accurate dataset.  
**Ongoing Monitoring** Continuously monitor the model’s performance to catch any new types of hallucinations that may emerge.  
**Cross-Reference** For critical applications, cross-reference the model’s outputs with verified information sources.  
**Domain-Specific Training** In fields like healthcare or law, involving experts in the fine-tuning process can help the model learn the subtleties and complexities of the domain, reducing the likelihood of generating incorrect information.  
**Scope Definition** Clearly define the scope of tasks that the model is designed to assist with and caution users against relying on it for tasks outside that scope.  

It’s important to remember that hallucinations cannot be completely eliminated. Therefore, it’s essential to be cautious of these limitations and have a human review and cross-reference the results when using LLMs for critical applications.

## Deploying Solutions to Mitigate Hallucination

### Retrieval-Augmented Generation
With RAG, a query comes into the knowledge base (which, in this case, is a vector database) as a semantic vector — a string of numbers.

The model then retrieves similar documents from the database using vector search, looking for documents whose vectors are close to the vector of the query.

Once the relevant documents have been retrieved, the query, along with these documents, is used by the LLM to summarize a response for the user. This way, the model doesn’t have to rely solely on its internal knowledge but can access whatever data you provide it at the right time. In a sense, it provides the LLM with “long-term memory” that it doesn’t possess on its own. The model can provide more accurate and contextually appropriate responses by including proprietary data stored in the vector database.

An alternate RAG approach incorporates fact-checking. The LLM is prompted for an answer, which is then fact-checked and reviewed against data in the vector database. An answer to the query is produced from the vector database, and then the LLM uses that answer as a prompt to discern whether it’s related to a fact.

### Reasoning
LLMs are good at a lot of things. They can predict the next word in a sentence, thanks to advances in “transformers,” which transform how machines understand human language by paying varying degrees of attention to different parts of the input data. LLMs are also good at boiling down a lot of information into a concise answer, and finding and extracting something you’re looking for from a large amount of text. Surprisingly, LLMS can also plan — they can gather data and plan a trip for you.

And maybe even more surprisingly, LLMs can use reasoning to produce an answer, in an almost human-like fashion. Because people can reason, they don’t need tons of data to make a prediction or decision. Reasoning also helps LLMs to avoid hallucinations. An example of this is “chain-of-thought prompting.”

This method helps models to break multistep problems into intermediate steps. With chain-of-thought prompting, LLMs can solve complex reasoning problems that standard prompt methods can’t (for an in-depth look, check out the blog post Language Models Perform Reasoning via Chain of Thought from Google).

If you give an LLM a complicated math problem, it might get it wrong. But if you provide the LLM with the problem as well as the method of solving it, it can produce an accurate answer — and share the reason behind the answer. A vector database is a key part of this method, as it provides examples of questions similar to this and populates the prompt with the example.

Even better, once you have the question and answer, you can store it in the vector database to further improve the accuracy and usefulness of your generative AI applications.

There are a host of other reasoning advancements you can learn about, including tree of thought, least to most, self-consistency and instruction tuning.

### Iterative Querying

The third method to help reduce LLM hallucinations is interactive querying. In this case, an AI agent mediates calls that move back and forth between an LLM and a vector database. This can happen multiple times iteratively in order to arrive at the best answer. An example of this is forward-looking active retrieval generation, also known as FLARE.

You take a question and then query your knowledge base for similar questions. You’d get a series of similar questions. Then you query the vector database with all the questions, summarize the answer, and check if the answer looks good and reasonable. If it doesn’t, repeat the steps until it does.

Other advanced interactive querying methods include AutoGPT, Microsoft Jarvis and Solo Performance Prompting.

There are many tools that can help you with agent orchestration. LangChain is a great example that helps you orchestrate calls between an LLM and a vector database. It essentially automates the majority of management tasks and interactions with LLMs and provides support for memory, vector-based similarity search, advanced prompt-templating abstraction and a wealth of other features. It also helps and supports advanced prompting techniques like chain-of-thought and FLARE.

Another such tool is CassIO, which was developed by DataStax as an abstraction on top of our Astra DB vector database, with the idea of making data and memory first-class citizens in generative AI. CassIO is a Python library that makes the integration of Cassandra with generative artificial intelligence and other machine learning workloads seamless by abstracting the process of accessing the database, including its vector search capabilities, and offering a set of ready-to-use tools that minimize the need for additional code.

# Conclusion





# Refrences
1. [3 Ways to Stop LLM Hallucinations](https://thenewstack.io/3-ways-to-stop-llm-hallucinations/#:~:text=LLMs%20hallucinate%20when%20they%20encounter,'t%20always%20be%20trusted).
2. [What are LLM Hallucinations?](https://www.iguazio.com/glossary/llm-hallucination/)