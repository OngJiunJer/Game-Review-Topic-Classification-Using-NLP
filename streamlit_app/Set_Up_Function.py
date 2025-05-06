import streamlit as st
import re
import nltk
from spellchecker import SpellChecker
import matplotlib.pyplot as plt
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.cluster import KMeans
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import learning_curve
from sklearn.metrics import classification_report
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image
import io
import google.generativeai as genai

# Make sure nltk packages are downloaded
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

# Initialize reusable tools
lemmatizer = WordNetLemmatizer() # Lemmatization Using WordNetLemmatizer and Pos Tag
spell_checker = SpellChecker() # For Speel Check
Stop_Word = set(stopwords.words('english')) # For Remove Stop Word

###############################################
# Set Up Function
###############################################

#Check the Distribution of Steam Review Length
def plot_review_length(df): 
    # Calculate review length
    df['review_length'] = df[df.columns[0]].apply(len)  # Apply len to each review in that column

    # Create the plot
    fig, ax = plt.subplots()
    ax.hist(df['review_length'], bins=100, edgecolor='black')
    ax.set_title('Distribution of Player Review Lengths')
    ax.set_xlabel('Review Length (Number of Characters)')
    ax.set_ylabel('Frequency')
    ax.set_xlim(0, 3200)

    # Save plot to image in memory
    image_bytes = io.BytesIO()
    fig.savefig(image_bytes, format='PNG')
    image_bytes.seek(0) 

    # Convert the image bytes to a PIL Image
    image = Image.open(image_bytes)

    # Display the plot in Streamlit
    st.pyplot(fig)

    # Send the image to Gemini for interpretation
    try:
        model = genai.GenerativeModel(model_name="gemini-2.0-flash")  # Replace with the appropriate model
        response = model.generate_content(["Please interpret this istribution of Player Review Lengths chart and "
        "explain with short and simple language explanation.", image])
        st.subheader("Chart Interpretation:")
        st.write(response.text)
    except Exception as e:
        st.error(f"Failed to interpret the image: {e}")

# Remove Missing Data Function
def remove_missing_data(df):
    df = df.dropna()
    return df

# Remove Duplicate Data Function
def remove_duplicate_data(df):
    df = df.drop_duplicates()
    return df

# Convert Review To Lowercase
def convert_lowercase(df):
    df['review_preprocessed'] = df['review_preprocessed'].str.lower()
    return df

# Remove Extra Space
def remove_extra_space(df):
    df['review_preprocessed'] = df['review_preprocessed'].str.strip()
    return df

#Filter Review Length
def filter_review_length(df, min_length, max_length):
    # Calculate review length & filter the review based on the input min and max length value
    df['review_preprocessed'] = df['review_preprocessed'].astype(str)
    df['review_length'] = df['review_preprocessed'].apply(len)
    df = df[(df['review_length'] >= min_length) & (df['review_length'] <= max_length)]
    df = df.drop('review_length', axis=1)
    return df

# Remove Numeric Text In Review
def remove_numeric_text(df):
    df['review_preprocessed'] = df['review_preprocessed'].str.replace(r'\d+', '', regex=True)
    return df

# Remove Emoji
def remove_emojis(text):
    emoji_pattern = re.compile("[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF"
                               "\U0001F680-\U0001F6FF\U0001F700-\U0001F77F"
                               "\U0001F780-\U0001F7FF\U0001F800-\U0001F8FF"
                               "\U0001F900-\U0001F9FF\U0001FA00-\U0001FA6F"
                               "\U0001FA70-\U0001FAFF\U00002600-\U000026FF"
                               "\U00002700-\U000027BF\U00002B50-\U00002B55"
                               "\U0001F1E6-\U0001F1FF]+", flags=re.UNICODE)
    return emoji_pattern.sub('', text)

#Handle Special Characters and Punctuation
def Remove_Special_Characters_and_Punctuation(Review):
    Review_Removed_Special_Characters_and_Punctuation = re.sub('\W+', ' ', Review)
    return Review_Removed_Special_Characters_and_Punctuation

# Handle Spelling Correction
def Check_Spelling(Review):
    words = Review.split()  # Split sentence into words
    corrected_words = [spell_checker.correction(word) or word for word in words]  # Correct each word
    return " ".join(corrected_words)  # Join words back into a sentence

# Remove Stop Word 
def Remove_Stop_Word(Review):
    Tokens = word_tokenize(Review)
    Review_Removed_Stop_Word = []
    for token in Tokens:
        if token not in Stop_Word:
            Review_Removed_Stop_Word.append(token)
    return " ".join(Review_Removed_Stop_Word)

#Get Word Pos Tag Function
def Get_Pos_Tag(word):
    tag = nltk.pos_tag([word])
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    Pos_Tag = tag_dict.get(tag[0][1][0].upper(), wordnet.NOUN)
    return Pos_Tag
    
#Lemmatization Function
def Lemmatization(Review):
    Tokens = word_tokenize(Review)
    Review_Lemmatize = []
    for token in Tokens:
        Review_Lemmatize.append(lemmatizer.lemmatize(token, pos = Get_Pos_Tag(token)))
    return " ".join(Review_Lemmatize)

# Elbow Method Function
def elbow_method(player_review_vec, max_k):
    inertia = []  # To store inertia value
    k_values = range(1, max_k + 1)

    # Compute KMeans for each K value
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(player_review_vec)

        # calculate inertia
        inertia.append(kmeans.inertia_)

    # Plot the inertia results
    fig, ax = plt.subplots()
    ax.plot(range(1, max_k + 1), inertia, marker='o')
    ax.set_title('Elbow Method for Optimal K')
    ax.set_xlabel('Number of Clusters (K)')
    ax.set_ylabel('Inertia')
    ax.set_xticks(range(1, max_k + 1))
    ax.grid(True)

    # Save plot to image in memory
    image_bytes = io.BytesIO()
    fig.savefig(image_bytes, format='PNG')
    image_bytes.seek(0) 

    # Convert the image bytes to a PIL Image
    image = Image.open(image_bytes)

    # Display the plot in Streamlit
    st.pyplot(fig)

    # Send the image to Gemini for interpretation
    try:
        model = genai.GenerativeModel(model_name="gemini-2.0-flash")  # Replace with the appropriate model
        response = model.generate_content(["Please interpret this elbow plot chart and explain with short and simple language explanation. "
        "And suggest what is the optimal number of cluster or k value", image])
        st.subheader("Chart Interpretation:")
        st.write(response.text)
    except Exception as e:
        st.error(f"Failed to interpret the image: {e}")

# Random Under Sampling
def random_under_sampling(x_train, y_train):
    under_sampler = RandomUnderSampler(sampling_strategy='auto', random_state=42)
    x_train_resampled, y_train_resampled = under_sampler.fit_resample(x_train, y_train)
    return x_train_resampled, y_train_resampled

# Plot Learning Curve Function
def plot_learning_curve(model, X, y):
    st.markdown("### Learning Curve")
    train_sizes, train_scores, test_scores = learning_curve(model, X, y, cv=5, scoring='accuracy')

    train_mean = train_scores.mean(axis=1)
    test_mean = test_scores.mean(axis=1)

    fig, ax = plt.subplots()
    ax.plot(train_sizes, train_mean, 'o-', color='blue', label='Training score')
    ax.plot(train_sizes, test_mean, 'o-', color='green', label='Cross-validation score')
    ax.set_title('Learning Curve - Logistic Regression')
    ax.set_xlabel('Training Set Size')
    ax.set_ylabel('Accuracy')
    ax.legend(loc='best')
    ax.grid(True)

    # Save plot to image in memory
    image_bytes = io.BytesIO()
    fig.savefig(image_bytes, format='PNG')
    image_bytes.seek(0) 

    # Convert the image bytes to a PIL Image
    image = Image.open(image_bytes)

    # Display the plot in Streamlit
    st.pyplot(fig)

    # Send the image to Gemini for interpretation
    try:
        model = genai.GenerativeModel(model_name="gemini-2.0-flash")  # Replace with the appropriate model
        response = model.generate_content(["Please interpret this learnning curve chart and explain with short and simple language explanation.", image])
        st.subheader("Chart Interpretation:")
        st.write(response.text)
    except Exception as e:
        st.error(f"Failed to interpret the image: {e}")

# Classification Report Function
def logistic_classification_report(model, x_test, y_test):
    # Predicting the test set results
    y_pred = model.predict(x_test)
    
    # Generate the classification report
    report = classification_report(y_test, y_pred)
    
    # Display the report on Streamlit
    st.markdown("### Classification Report")
    st.text(report)

    # Send the classification report to Gemini for interpretation
    try:
        # Instantiate the model (you may already have this from a previous part of your code)
        model = genai.GenerativeModel(model_name="gemini-2.0-flash")  # Use the appropriate Gemini model
        
        # Send the classification report text for interpretation
        response = model.generate_content([f"Please interpret this classification report and explain with short and simple language explanation.\n{report}"])
        
        # Display Gemini's response
        st.subheader("Gemini's Interpretation of the Classification Report:")
        st.write(response.text)

    except Exception as e:
        st.error(f"Failed to interpret the report: {e}")


# Confusion Matrix Plot
def plot_logistic_confusion_matrix(model, x_test, y_test):
    y_pred = model.predict(x_test)

    st.markdown("### Confusion Matrix")
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax)
    ax.set_title("Confusion Matrix - Logistic Regression")

    # Save plot to image in memory
    image_bytes = io.BytesIO()
    fig.savefig(image_bytes, format='PNG')
    image_bytes.seek(0) 

    # Convert the image bytes to a PIL Image
    image = Image.open(image_bytes)

    # Display the plot in Streamlit
    st.pyplot(fig)

    # Send the image to Gemini for interpretation
    try:
        model = genai.GenerativeModel(model_name="gemini-2.0-flash")  # Replace with the appropriate model
        response = model.generate_content(["Please interpret this confusion matrix chart and explain with short and simple language explanation.", image])
        st.subheader("Chart Interpretation:")
        st.write(response.text)
    except Exception as e:
        st.error(f"Failed to interpret the image: {e}")

# Setup Button Setting
def inject_button_style():
    st.markdown("""
        <style>
        div.stButton > button {
            background-color: white;     /* 🔴 Button background is white */
            color: black;                /* 🔤 Button text color is black for contrast */
            border-radius: 8px;          /* 🔄 Rounded corners */
            padding: 10px 24px;          /* 📏 Button size */
            font-size: 16px;             /* 🔠 Font size */
            border: 2px solid #ddd;      /* Optional: Add a light border for better visibility */
        }

        div.stButton > button:hover {
            background-color: #f1f1f1;  /* 🟠 Hover effect: Light gray on hover */
            color: black;               /* Text remains black on hover */
        }
        </style>
    """, unsafe_allow_html=True)