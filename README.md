# Game-Review-Classification-Using-NLP
Built and deployed an NLP-based tool to classify game reviews using BERT variants (Tiny, Mini,  Small, Medium), Convolutional Neural Network (CNN), Long Short-Term Memory (LSTM), Naive  Bayes, and Logistic Regression. Achieved 97% accuracy with Logistic Regression and enabled  real-time sentiment analysis via a Streamlit web app.

# Clustering Code Procedure
- This section describes the process carried out in the "Clustering_Code.ipynb" file.
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
- Remove Missing Data
- Remove Duplicate Data
- Convert Player Review to Lowercase
- Remove Unwanted Column
- Remove Extra Whitespace
- Filter Review Length Between 20-150
- Remove Numeric Data in Review
- Remove Emoji
- Sampling 100K Review
- Remove Special Characters and Punctuation
- Spelling Correction
- Remove Stop Words (Not Required When Building Bert Model)
- Lemmatization (Not Required When Building Bert Model)

## Step 3: Data Preprocessing
- Remove Missing Data
