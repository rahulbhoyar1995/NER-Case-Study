# Named Entity Recognition (NER) Project - Case Study
### Author : Rahul Bhoyar
### Date : 17/06/2024

### About the Dataset 
The dataset (ner_dataset.csv) maps sentence numbers to words, each associated with a respective Part of Speech (POS) tag and Named Entity Recognition (NER) tag. For this case study, we will focus on the word and its NER tag, ignoring the POS tag.


### Sequence Tagging Scheme: IOB2

I (Inside): The word is inside a chunk.

O (Outside): The word belongs to no chunk.

B (Beginning): The word is the beginning of a chunk.

### Columns
Sentences #: Sentence number.

Word: Word to be classified.

POS: Part of Speech tag for the respective word (ignored).

Tag: NER tag for the respective word.

### Probable Tasks
1. Divide the dataset into 3 parts: Train, Validation, Test (at least 20%)

2. Identify the matrices for evaluating model's performance.

3. Pre-process the data such that words of each sentence are mapped to their respective NER tags.

4. Develop a baseline model which takes a sentence (list of words) as input and predicts NER tag for each word in that sentence.

5. Identify the shortcomings of the baseline model.

6. Develop a new model which overcomes the shortcomings of the baseline model.

7. Identify future scope to further optimize the model.

### System Design Tasks

1. Design system architecture to deploy ML Model in production.
   
2. Perform a canary build.
   
3. Strategy for ML Model Monitoring.
   
4. Perform load and stress testing.
   
5. Track, monitor, and audit ML training.
   
6. Design framework for continuous delivery and automation of machine learning tasks.
   
### Future Optimization Scope

1. Use Pre-trained Embeddings: Incorporate GloVe or BERT embeddings to improve performance.
   
2. Hyperparameter Tuning: Experiment with different hyperparameters like batch size, learning rate, number of LSTM units, etc.
   
3. Ensemble Methods: Combine predictions from multiple models to improve accuracy.
   
4. Error Analysis: Analyze errors to understand common failure cases and address them.
   
### Theoretical Aspects of the Algorithm
   
The Bi-LSTM (Bidirectional Long Short-Term Memory) model is used for the NER task. The steps involved are:

(a) Preprocessing:

Tokenization of words.
Padding sequences to a uniform length (max_len = 50).

(b) Model Architecture:

Input layer with pre-trained word embeddings.
Bidirectional LSTM layers to capture context from both directions.
Dense layer with softmax activation for outputting NER tags.

(c) Training:

Model is trained on the training dataset.
Validation dataset is used to tune hyperparameters and prevent overfitting.
Test dataset is used to evaluate the final model performance.

(d) Evaluation Metrics:

Accuracy: Measures the fraction of correct predictions.
Loss: Measures the error between predicted and actual NER tags.


### Model Deployment and Monitoring
(a) System Architecture:

Data ingestion

Feature store

Model serving

Monitoring and logging

(b) Canary Build:

Deploying the model to a subset of users

Monitoring performance

Gradually rolling out to all users if no issues are found

(c) Model Monitoring:

Tracking performance metrics (accuracy, latency)

Setting up alerts for significant deviations

Regular retraining and updating of the model

(d) Load and Stress Testing:

Simulating high traffic to ensure system stability
Using tools like Apache JMeter and Locust for testing

(e) Continuous Delivery and Automation:

CI/CD pipelines for automated model deployment
Automated testing and validation steps
Continuous monitoring and retraining framework
