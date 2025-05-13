import streamlit as st
from PIL import Image
import google.generativeai as genai
import io

# Configure API key (Need to put your own gemini API)
genai.configure(api_key="")

# Load Gemini model
model = genai.GenerativeModel(model_name="gemini-2.0-flash")

def show():
    st.title("Chart Interpreter with Gemini (Gemini-Pro-Vision)")

    st.sidebar.header("How to Use:")
    st.sidebar.write("1. Upload an image of a chart.")
    st.sidebar.write("2. Or ask a question related to the chart or any data-related doubt.")
    st.sidebar.write("3. Click the submit button to get the result.")

    # Upload Chart Image
    uploaded_file = st.file_uploader("Upload an Image (Chart)", type=["jpg", "png", "jpeg"])

    # Manual question input
    manual_text = st.text_area("Or, ask a question about the chart or your data:")

    # Button to submit question
    submit_question = st.button("Submit Question")

    # --- If image is uploaded ---
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Chart", use_column_width=True)

        image_bytes = io.BytesIO()
        image.save(image_bytes, format="PNG")
        image_bytes.seek(0)

        st.write("Interpreting the chart...")

        try:
            response = model.generate_content(["Please interpret this chart.", image])
            st.subheader("Chart Interpretation:")
            st.write(response.text)
        except Exception as e:
            st.error(f"Failed to interpret the image: {e}")

    # --- If question is entered and button is clicked ---
    elif submit_question and manual_text.strip() != "":
        st.write("Processing your question...")
        try:
            response = model.generate_content(manual_text)
            st.subheader("Answer:")
            st.write(response.text)
        except Exception as e:
            st.error(f"Failed to process the question: {e}")

    # --- No input provided ---
    elif not uploaded_file and not manual_text:
        st.write("Please upload a chart or ask a question to begin.")
