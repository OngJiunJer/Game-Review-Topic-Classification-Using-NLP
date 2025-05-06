import streamlit as st

def show():
    st.title("🎮 Player Review Classifier App")

    st.markdown("""
    Welcome to the Player Review Classifier App!  
    You can choose powerful features using the sidebar:

    1. 🚀 **Use Pre-Built Model Classify Review (Logistic Regression)** – Quickly classify player reviews using our pre-trained logistic regression model.
    2. 🛠️ **Customize Your Own Model** – Build and train your own logistic regression model to classify reviews which are not limited to player review.  
    3. 📤 **Upload Custom Model to Classify Review** – Use a logistic regression model you've previously trained and saved to classify new reviews.  
    4. 🤖 **Chat with Gemini API** – Engage with the Gemini API by asking any question or uploading a chart or plot to help interpret the data.
    """)