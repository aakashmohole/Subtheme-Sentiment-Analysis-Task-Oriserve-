# Subthem Sentiment Analysis Task

## Overview
This project aims to develop a sentiment analysis model using deep learning techniques to predict the sentiment (positive or negative) of textual reviews. The model is trained on a dataset containing labeled reviews and deployed as a web application using gradio.

## Table of Contents
- [Overview](#overview)
- [Data Collection](#data-collection)
- [Data Preprocessing](#data-preprocessing)
- [Model Development](#model-development)
- [Model Evaluation](#model-evaluation)
- [Deployment](#deployment)
- [Usage](#usage)
- [Conclusion](#conclusion)
- [References](#references)

## Data Collection
![Loading Data](https://github.com/aakashmohole/Subtheme-Sentiment-Analysis-Task-Oriserve-/blob/main/images/loaddata.png)
- The dataset used for training is collected from link given in mail.
- It contains reviews labeled with positive or negative sentiments.

## Data Preprocessing
![Data Overall view](https://github.com/aakashmohole/Subtheme-Sentiment-Analysis-Task-Oriserve-/blob/main/images/count.png)
- Tokenization: Convert text data into tokens (words or subwords).
- Padding: Ensure that all input sequences have the same length.
- Encoding: Convert tokens into numerical representations.
- Splitting: Divide the dataset into training and testing sets.
 ![Data after processing](https://github.com/aakashmohole/Subtheme-Sentiment-Analysis-Task-Oriserve-/blob/main/images/count%202.png)

## Model Development
![Model Building](https://github.com/aakashmohole/Subtheme-Sentiment-Analysis-Task-Oriserve-/blob/main/images/train.png)
- We use a Recurrent Neural Network (RNN) with LSTM cells for sentiment analysis.
- The model is defined using TensorFlow/Keras.

## Model Training
![Model Training](https://github.com/aakashmohole/Subtheme-Sentiment-Analysis-Task-Oriserve-/blob/main/images/rrain2.png)
- The model is compiled with a binary cross-entropy loss function and Adam optimizer.
- It is trained on the training dataset with validation on the testing dataset.

## Model Evaluation
![Training Graph](https://github.com/aakashmohole/Subtheme-Sentiment-Analysis-Task-Oriserve-/blob/main/images/training%20graph.png)
- The performance of the model is evaluated using accuracy, precision, recall, and F1-score.
- The training and validation accuracy/loss are plotted over epochs.

## Deployment
![Deployment](https://github.com/aakashmohole/Subtheme-Sentiment-Analysis-Task-Oriserve-/blob/main/images/output.png)
- The trained model is deployed as a web application using Gradio.
- Users can input text and get sentiment predictions through a user-friendly interface.

## Usage
To run the Gradio app locally:

1. Clone the repository:
    ```bash
    https://github.com/aakashmohole/Subtheme-Sentiment-Analysis-Task-Oriserve-.git
    ```

2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

3. Run the Gradio app:
    ```bash
    python app.py
    ```
4. Open your web browser and navigate to the provided URL to use the app.


## Conclusion
- The sentiment analysis model successfully predicts the sentiment of textual reviews.
- The web application provides an easy-to-use interface for sentiment prediction.

## References
- TensorFlow: [https://www.tensorflow.org/](https://www.tensorflow.org/)
- Keras: [https://keras.io/](https://keras.io/)
- Gradio: [https://gradio.app/](https://gradio.app/)

---


