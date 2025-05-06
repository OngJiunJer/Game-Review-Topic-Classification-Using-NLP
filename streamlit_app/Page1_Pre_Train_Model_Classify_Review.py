import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import Set_Up_Function

# Initialize step states
for step in ["Page1_Step1", "Page1_Step2", "Page1_Step3", "Page1_Step4"]:
    if step not in st.session_state:
        st.session_state[step] = False

# Reset step states
def reset_session_state():
    # List of steps to reset
    steps = ["Page1_Step1", "Page1_Step2", "Page1_Step3", "Page1_Step4"]
    
    # Loop through each step
    for step in steps:
        if step in st.session_state:
            st.session_state[step] = False  # Reset the step to False

# Show All The Page 1 Classify Review Interface
def show():
    ###############################################
    # Button Setting
    ###############################################
    Set_Up_Function.inject_button_style()

    ###############################################
    # Main Title
    ###############################################
    st.title("🚀 Use Pre-Built Model Classify Review (Logistic Regression)")

    # Add a horizontal line for separation
    st.markdown("---")

    ###############################################
    # Step1: Upload Dataset Section
    ###############################################
    # Description of Step 1
    st.header("Step1: Upload Your Dataset")
    # Display the markdown instructions
    st.markdown("""
    ### 🧾 Upload Instructions: What Kind of Dataset Can You Import?

    You can upload **two types of datasets** into this app:

    ---

    #### 📄 1. One-Column Dataset
    This dataset contains only **one column**, usually raw text or reviews that have **not been preprocessed yet**.

    **Example:**
                
    ✅ **What the system will do:**  
    If you upload this type, the app will give the access to Step 2: Review Length Plot to proceed.

    ---

    #### 🧹 2. Preprocessed Dataset
    This dataset must contains **review_preprocessed** column:

    **Example:**
                
    ✅ **What the system will do:**  
    If the column `review_preprocessed` is detected, the system will give the acces for you to skip Step 2: Review Length Plot and Step 3: Preprocessing Your Dataset

    ---

    ### ⚠️ Notes
    - The `review_preprocessed` column name **must be exact** (case-sensitive).
    - Extra columns (besides the two needed) will be ignored by default.
    """)

    # File uploader with key
    uploaded_file = st.file_uploader("Upload your dataset", type=["csv"], key="uploaded_file")

    # Upload Dataset Button
    if st.button("Upload Dataset"):
            
        # Reset everything before processing new file
        reset_session_state()

        # Now proceed with processing
        df = pd.read_csv(uploaded_file)

        # 1. One-column dataset
        if df.shape[1] == 1:
            if isinstance(df.iloc[0, 0], str):
                st.success("File uploaded successfully! (1. One-Column Dataset)")
                df[df.columns[0]] = df[df.columns[0]].astype(str) # Ensure the review text column is string
                st.session_state.Page1_Step1 = True
                st.session_state.df = df # Save df in session state
                st.dataframe(df.head())
            else:
                st.error("The uploaded file contains non-text data. Please upload a file where the column contains review text.")

        # 2. Preprocessed dataset
        elif 'review_preprocessed' in df.columns:
            if isinstance(df['review_preprocessed'].iloc[0], str):
                st.success("File uploaded successfully! (2. Preprocessed Dataset)")
                df['review_preprocessed'] = df['review_preprocessed'].astype(str) # Ensure the review text column is string
                st.session_state.Page1_Step1 = True
                st.session_state.Page1_Step3 = True
                st.session_state.df = df # Save df in session state
                st.dataframe(df.head())
            else:
                st.error("The uploaded file contains non-text data. Please upload a file where the column contains review text.")

        # 3. Did not upload dataset
        elif uploaded_file is None:
            st.error("Please upload a CSV dataset.")

        # 4. Doesn't meet criteria
        else:
            st.error("The uploaded file did not fulfill the requirement to upload.")
    
        
    # Add a horizontal line for separation
    st.markdown("---")

    ###############################################
    # Step2: Review Length Plot Title
    ###############################################
    # Description of Step 2
    st.header("Step2: Review Length Distribution")
    st.markdown("Review length plot can help to understand the distribution review length of your dataset. Then you can decide the min and max word length of your review.")
    # Plot Review Length Button

    if st.session_state.Page1_Step1:
        if st.button("Plot"):
            with st.spinner("Plotting..."):
                # Retrieve the dataframe from session state
                df = st.session_state.get('df', None)
                
                # Plot Review Length Chart
                Set_Up_Function.plot_review_length(df)

                # Step 2 Completed
                st.session_state.Page1_Step2 = True
    else:
        st.warning("Run Step 1 First")

    # Set Up The Min & Max Value
    st.subheader("Select Review Length Range")
    st.markdown("Input your preferred min and max length.")
    # Input Min Length
    min_length = st.number_input("Enter min (20-1000)", min_value=20, max_value=1000, value=20)
    st.write("You entered min:", min_length)

    # Input Max Length
    max_length = st.number_input("Enter max (150-3000)", min_value=150, max_value=3000, value=150)
    st.write("You entered max:", max_length)

    # Check if min_length is greater than max_length
    if min_length >= max_length:
        st.error("Min length must be smaller than max length. Please adjust your inputs.")
    else:
        st.success("The min length is less than the max length. You are good to go!")

    # Add a horizontal line for separation
    st.markdown("---")

    ###############################################
    # Step3: Preprocessing Section
    ###############################################
    # Description of Step 3
    st.header("Step3: Preprocessing Your Dataset")
    st.markdown("### 🧹 Preprocessing Tasks")
    st.markdown("1. **Remove Missing Data** - Removes rows with null values.")
    st.markdown("2. **Remove Duplicates** - Eliminates duplicate entries in the dataset.")
    st.markdown("3. **Convert to Lowercase** - Converts all review text to lowercase.")
    st.markdown("4. **Remove Extra Spaces** - Trims unnecessary white spaces from text.")
    st.markdown("5. **Remove Numeric Text** - Deletes numeric characters from the text.")
    st.markdown("6. **Filter by Review Length** - Keeps reviews within specified length boundaries.")
    st.markdown("7. **Remove Emojis** - Deletes emoji characters from the text.")
    st.markdown("8. **Remove Special Characters and Punctuation** - Cleans symbols and punctuation.")
    st.markdown("9. **Check Spelling** - Corrects misspelled words (⚠️ may be slow on large datasets).")
    st.markdown("10. **Remove Stop Word** - Remove all the stop words such as *the*, *is*, *at*, and so on.")
    st.markdown("11. **Lemmatization** - Changes words to their root form")

    # Add checkbox for spelling correction
    do_spell_check = st.checkbox("Include Check Spelling (Time-Costly)")

    # Run ALL Preprocessing Button
    if st.session_state.Page1_Step2:
        if st.button("Run ALL Preprocessing"):
            with st.spinner("Preprocessing..."):
                # Retrieve the dataframe from session state
                df = st.session_state.get('df', None)

                # Duplicate the only column to a new column named 'review_temp'
                original_review_column = df.columns[0]
                df['review_preprocessed'] = df[original_review_column]

                # Run Preprocessing Task One by One
                df = Set_Up_Function.remove_missing_data(df)
                st.write("Missing data removed.")
                df = Set_Up_Function.remove_duplicate_data(df)
                st.write("Duplicates removed.")
                df = Set_Up_Function.convert_lowercase(df)
                st.write("Text converted to lowercase.")
                df = Set_Up_Function.remove_extra_space(df)
                st.write("Extra spaces removed.")
                df = Set_Up_Function.remove_numeric_text(df)
                st.write("Numeric text removed.")
                df = Set_Up_Function.filter_review_length(df, min_length, max_length)
                st.write("Reviews filtered by length.")
                df['review_preprocessed'] = df['review_preprocessed'].apply(Set_Up_Function.remove_emojis)
                st.write("Emojis removed.")
                df['review_preprocessed'] = df['review_preprocessed'].apply(Set_Up_Function.Remove_Special_Characters_and_Punctuation)
                st.write("Special characters and punctuation removed.")

                # Optional spelling correction
                if do_spell_check:
                    df['review_preprocessed'] = df['review_preprocessed'].apply(Set_Up_Function.Check_Spelling)
                    st.write("✅ Spelling checked.")
                else:
                    st.write("❌ Skipped spelling correction.")

                df['review_preprocessed'] = df['review_preprocessed'].apply(Set_Up_Function.Remove_Stop_Word)
                st.write("Stop word removed.")
                df['review_preprocessed'] = df['review_preprocessed'].apply(Set_Up_Function.Lemmatization)
                st.write("Lemmatization Done.")

                # Store the preprocessed dataframe in session state
                st.session_state.df = df

                st.success("Preprocessing complete!")
                st.dataframe(df)

                # Step 3 Completed
                st.session_state.Page1_Step3 = True
    else:
        st.warning("Run Step 2 First")

    # Add a horizontal line for separation
    st.markdown("---")

    ###############################################
    # Download Pre-Processed Dataset Section (Optional)
    ###############################################
    st.header("Download Pre-Processed Dataset (Optional)")
    st.markdown("This step is to download the preprocessed dataset. So that you will not require repeated preprocessed the same dataset.")
    # Input File Name
    pre_file_name  = st.text_input("Enter File Name", value="Preprocessed_Reviews.csv")
    st.write("You entered file name:", pre_file_name )
    # Show download button directly if Step3 is completed
    if st.session_state.Page1_Step3:
        # Retrieve the dataframe from session state
        df = st.session_state.get('df', None)

        # Convert the dataframe to CSV
        csv = df.to_csv(index=False).encode('utf-8')
        
        # Download button (only one click needed)
        st.download_button(
            label="Download Pre-Processed CSV",  # Text for the button
            data=csv,  # The CSV file data
            file_name=pre_file_name,  # The name of the file when downloaded
            mime='text/csv'  # The MIME type for CSV files
        )
    else:
        st.warning("Run Step 3 First")

    # Add a horizontal line for separation
    st.markdown("---")

    ###############################################
    # Step4: Classification Section
    ###############################################
    # Description of Step 4
    st.header("Step4: Classify Player Reviews")
    st.markdown("Topic 0: Positive Vibes")
    st.caption("Reviews in this cluster express general enjoyment and satisfaction with the game, reflecting a strong positive emotional response from players.")
    st.markdown("Topic 1: Worth Buying")
    st.caption("This group highlights the game's value for money and overall appeal, with players often suggesting it’s a good purchase decision.")
    st.markdown("Topic 2: Story & Graphics")
    st.caption("Players in this cluster are drawn to the narrative and visual aspects, showing appreciation for the game's storytelling and artistic design.")
    st.markdown("Topic 3: Well-Rounded")
    st.caption("These reviews offer balanced praise across multiple aspects of the game, indicating a well-crafted experience that delivers on various fronts.")
    st.markdown("Topic 4: All-Time Best")
    st.caption("Contains highly enthusiastic reviews from players who consider the game among the best they’ve played, often describing it as a top-tier experience.")
    st.markdown("Topic 5: Top Picks")
    st.caption("Strong recommendations and standout feedback place this game as a favorite choice among players in this cluster.")
    st.markdown("Topic 6: Long Sessions")
    st.caption("Reviews here reflect deep engagement, with players spending extended periods immersed in the gameplay, indicating strong retention.")
    st.markdown("Topic 7: Multiplayer Fun")
    st.caption("This cluster centers around the social and cooperative aspects of the game, with players enjoying the experience with others.")

    # Run Classification Button
    if st.session_state.Page1_Step3:
        if st.button("Run Classification"):
            with st.spinner("Classifying..."):
                # Retrieve the dataframe from session state
                df = st.session_state.get('df', None)

                # Load LR Model & Vectorizer
                lr_model = joblib.load("logistic_regression_model.pkl")
                TFvectorizer = joblib.load("TFvectorizer.pkl")

                # Use the loaded vectorizer to vectorize the preprocessed review
                df['review_preprocessed'] = df['review_preprocessed'].astype(str).tolist()
                X_tfidf = TFvectorizer.transform(df['review_preprocessed'])

                # Predict using the loaded LR model
                predictions = lr_model.predict(X_tfidf)

                # Add predictions to DataFrame
                df['cluster'] = predictions

                # Define cluster_topics to store topic name
                cluster_topics = {
                    0: "Positive Vibes",
                    1: "Worth Buying",
                    2: "Story & Graphics",
                    3: "Well-Rounded",
                    4: "All-Time Best",
                    5: "Top Picks",
                    6: "Long Sessions",
                    7: "Multiplayer Fun"
                }
                # Create new 'Topic' column based on cluster_topics dictionary
                df['Topic'] = df['cluster'].map(cluster_topics)

                st.success("Classification complete!")
                st.write("Here is a preview of the classified dataset:")
                st.dataframe(df)

                # Store the classified dataframe in session state
                st.session_state.df = df

                # Step 4 Compledted
                st.session_state.Page1_Step4 = True
    else:
        st.warning("Run Step 3 First.")

    # Add a horizontal line for separation
    st.markdown("---")

    ###############################################
    # Step5: Show Class Distribution
    ###############################################
    st.header("Step5: Show Class Distribution")
    st.markdown("This step is to reveal the frequency of each topic.")
    if st.session_state.Page1_Step4:
        if st.button("Predicted Class Distribution"):
            # Retrieve the dataframe from session state
            df = st.session_state.get('df', None)

            # Plot Class Distribution
            fig, ax = plt.subplots(figsize=(15, 10))
            sns.countplot(data=df, x='Topic', ax=ax, palette='Set2')
            # Add count values
            for bar in ax.patches:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2,
                        height + 0.5,
                        str(int(height)),
                        ha='center')
            st.pyplot(fig)
    else:
        st.warning("Run Step 4 First")

    st.markdown("---")

    ###############################################
    # Step6: Download Classified Dataset Section
    ###############################################
    st.header("Step6: Download Classified Dataset")
    st.markdown("This step is to dowload the classified dataset.")
    # Input File Name
    class_file_name  = st.text_input("Enter File Name", value="Classified_Reviews.csv")
    st.write("You entered file name:", class_file_name )

    # Show download button directly if Step4 is completed
    if st.session_state.Page1_Step4:
        # Retrieve the dataframe from session state
        df = st.session_state.get('df', None)

        # Convert the dataframe to CSV
        csv = df.to_csv(index=False).encode('utf-8')
        
        # Download button (only one click needed)
        st.download_button(
            label="Download Classified CSV",  # Text for the button
            data=csv,  # The CSV file data
            file_name=class_file_name,  # The name of the file when downloaded
            mime='text/csv'  # The MIME type for CSV files
        )
    else:
        st.warning("Run Step 4 First")

    # Add a horizontal line for separation
    st.markdown("---")



# when presentation use large dataset to shows high accuracy result, and small dataset to shows the flow