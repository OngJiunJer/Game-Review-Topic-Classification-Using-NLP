# Game-Review-Classification-Using-NLP
Built and deployed an NLP-based tool to classify game reviews using BERT variants (Tiny, Mini,  Small, Medium), Convolutional Neural Network (CNN), Long Short-Term Memory (LSTM), Naive  Bayes, and Logistic Regression. Achieved 97% accuracy with Logistic Regression and enabled  real-time sentiment analysis via a Streamlit web app.

---

# Clustering Code Procedure
- This section describes the process in the "Clustering_Code.ipynb" file.
- It is the first file to be executed, as the original dataset does not include a target variable.
- Clustering techniques are applied to the Steam reviews to group similar reviews into multiple clusters.
- These clusters are then used to assign labels to the data (target variable), making it suitable for supervised learning in later steps.

## Step 1: Data Understanding
- The dataset is called "Steam Reviews".
- It was obtained from Kaggle (https://www.kaggle.com/datasets/andrewmvd/steam-reviews).
- Loaded and inspected the dataset.
- Check Missing & Duplicate Data.
- Visualized Steam Review distributions.

## Step 2: Data Preprocessing
- Remove Missing Data.
- Remove Duplicate Data.
- Convert Player Review to Lowercase.
- Remove Unwanted Column.
- Remove Extra Whitespace.
- Filter Review Length Between 20-150.
- Remove Numeric Data in Review.
- Remove Emoji.
- Sampling 100K Review.
- Remove Special Characters and Punctuation.
- Spelling Correction.
- Remove Stop Words (Not Required When Building Bert Model).
- Lemmatization (Not Required When Building Bert Model).

## Step 3: Build K-Means Clustering Model
- Apply the Elbow Method to determine the optimal number of clusters (K value).
- Train the K-Means clustering model using the selected K value on the cleaned dataset.
- Retrieve the top ten keywords for each cluster to serve as descriptive labels, helping to interpret the themes or topics represented in each group.

## Step 4: Export the Pre-processed Dataset
- Extract the pre-processed dataset, which will be used to train the Bert, Deep Learning, and Machine Learning Model.

---

# Build Bert Model Code Procedure
- This section describes the process outlined in the "Build_Bert_Model_Code.ipynb" file.
- The pre-processed dataset will be used to build four different types of BERT models: Tiny, Mini, Small, and Medium BERT.

## Step 1: Import dataset
- Import the pre-processed dataset from the Clustering_Code.ipynb file.

## Step 2: Data Preprocessing
- While most preprocessing is handled in the "Clustering_Code.ipynb" file, several additional steps are required before building the BERT models:
  1) Remove unwanted columns.
  2) Ensure the Steam review column (review_text) is of string data type.
  3) Split the data into 80% training and 20% testing subsets.
  4) Apply Random Under-Sampling to address class imbalance.
  5) Convert the DataFrame into a Hugging Face Dataset format.

## Step 3: Build Bert Model 
- Before building the BERT models, it is necessary to load the appropriate BERT tokenizer for text tokenization and the corresponding BERT model for training.
- Begin building and training the following four variants of the BERT model:
  1) Tiny Bert.
  2) Mini Bert.
  3) Small Bert.
  4) Medium Bert.
 
## Step 4: Evaluation
- Three types of evaluation methods are used to assess the performance of the BERT models:
  1) Learning Curve
  2) Classification Report
  3) Confusion Matrix
  
---

# Build ML & DL Model Code Procedure
- This section outlines the process described in the "Build_ML_&_DL_Model_Code.ipynb" file.
- The pre-processed dataset is used to build four different types of machine learning and deep learning models: CNN, LSTM, Naive Bayes, and Logistic Regression.

## Step 1: Import dataset
- Import the pre-processed dataset from the Clustering_Code.ipynb file.

## Step 2: Data Preprocessing
- While most preprocessing is handled in the "Clustering_Code.ipynb" file, several additional steps are required before building the ML & DL models:
  1) Remove unwanted columns.
  2) Ensure the Steam review column (review_text) is of string data type.
  3) Apply tokenization (two approaches depending on ML or DL model).
  4) Split the data into 80% training and 20% testing subsets.
  5) Apply Random Under-Sampling to address class imbalance.

## Step 3: Build Bert Model 
- Begin building and training four different variants of the ML & DL model:
  1) CNN
  2) LTSM
  3) Naive Bayes
  4) Logistic Regression
 
## Step 4: Evaluation
- Three types of evaluation methods are used to assess the performance of the ML & DL models:
  1) Learning Curve
  2) Classification Report
  3) Confusion Matrix

---

# Streamlit Deployment
- The "Streamlit_app" file contains all the code needed to deploy the application, which includes four different features:
  1) Use Pre-Built Model Classify Review (Logistic Regression)
  2) Customize Your Own Model
  3) Upload Custom Model to Classify Review
  4) Chat with the Gemini API
