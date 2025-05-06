import streamlit as st
import Home
import Page1_Pre_Train_Model_Classify_Review
import Page2_Customize_Model
import Page3_Upload_Customize_Model
import Page4_Gemini_API

#python -m streamlit run Main.py

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to", 
    (
        "Home",
        "🚀 Use Pre-Built Model Classify Review (Logistic Regression)",
        "🛠️ Customize Your Own Model",
        "📥 Upload Custom Model to Classify Review",
        "🤖 Chat with Gemini API"
    )
)

# Load the selected page
if page == "Home":
    Home.show()
elif page == "🚀 Use Pre-Built Model Classify Review (Logistic Regression)":
    Page1_Pre_Train_Model_Classify_Review.show()
elif page == "🛠️ Customize Your Own Model":
    Page2_Customize_Model.show()
elif page == "📥 Upload Custom Model to Classify Review":
    Page3_Upload_Customize_Model.show()
elif page == "🤖 Chat with Gemini API": 
    Page4_Gemini_API.show()
