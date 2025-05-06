import streamlit as st
import pandas as pd
import joblib
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import numpy as np
import Set_Up_Function

# Initialize step states
for step in ["Page2_Step1", "Page2_Step2", "Page2_Step3", "Page2_Step4", "Page2_Step5", "Page2_Step6", "Page2_Step7", "Page2_Step8"]:
    if step not in st.session_state:
        st.session_state[step] = False

# Reset step states
def reset_session_state():
    # List of steps to reset
    steps = ["Page2_Step1", "Page2_Step2", "Page2_Step3", "Page2_Step4", "Page2_Step5", "Page2_Step6", "Page2_Step7", "Page2_Step8"]
    
    # Loop through each step
    for step in steps:
        if step in st.session_state:
            st.session_state[step] = False  # Reset the step to False

# Show All The Page 2 Customize Model
def show():
    ###############################################
    # Button Setting
    ###############################################
    Set_Up_Function.inject_button_style()

    ###############################################
    # Main Title
    ###############################################
    st.title("🛠️ Customize Your Own Model")

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
                st.session_state.Page2_Step1 = True
                st.session_state.df = df # Save df in session state
                st.dataframe(df.head())
            else:
                st.error("The uploaded file contains non-text data. Please upload a file where the column contains review text.")

        # 2. Preprocessed dataset
        elif 'review_preprocessed' in df.columns:
            if isinstance(df['review_preprocessed'].iloc[0], str):
                st.success("File uploaded successfully! (2. Preprocessed Dataset)")
                df['review_preprocessed'] = df['review_preprocessed'].astype(str) # Ensure the review text column is string
                st.session_state.Page2_Step1 = True
                st.session_state.Page2_Step3 = True
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

    if st.session_state.Page2_Step1:
        if st.button("Plot"):
            with st.spinner("Plotting..."):
                # Retrieve the dataframe from session state
                df = st.session_state.get('df', None)
                
                # Plot Review Length Chart
                Set_Up_Function.plot_review_length(df)

                # Step 2 Completed
                st.session_state.Page2_Step2 = True
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
    if st.session_state.Page2_Step2:
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
                st.session_state.Page2_Step3 = True
    else:
        st.warning("Run Step 2 First")

    # Add a horizontal line for separation
    st.markdown("---")

    ###############################################
    # Download Pre-Processed Dataset Section (Optional)
    ###############################################
    st.header("Download Pre-Processed Dataset (Optional)")
    # Input File Name
    pre_file_name  = st.text_input("Enter File Name", value="Preprocessed_Reviews.csv")
    st.write("You entered file name:", pre_file_name )
    # Show download button directly if Step3 is completed
    if st.session_state.Page2_Step3:
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
    # Step4: Plot Elbow & Choosing the Number of Clusters Section
    ###############################################
    st.header("Step4: Plot Elbow & Choosing the Number of Clusters")
    st.markdown("In this section, you'll interpret the elbow plot to determine the ideal number of clusters — " \
    "which essentially represents the number of topics you want to identify.")
    st.markdown("Look at the elbow plot and choose the number of clusters (k) at the point where the curve starts to flatten, " \
    "as this indicates the optimal number of topics to classify.")

    if st.session_state.Page2_Step3:
        if st.button("Plot Elbow"):
            # Retrieve the dataframe from session state
            df = st.session_state.get('df', None)

            # Load vectorizer
            TFvectorizer = joblib.load("TFvectorizer.pkl")

            # Transfer the player review in to numeric form
            player_review_overall_vectorize = TFvectorizer.fit_transform(df['review_preprocessed'])

            # Run elbow_method function
            Set_Up_Function.elbow_method(player_review_overall_vectorize, 20)  

            # Step 4 Completed
            st.session_state.Page2_Step4 = True 
    else:
        st.warning("Run Step 3 First")

    k = st.number_input("Enter the number of clusters (k) (2-20)", min_value=2, max_value=20, value=2)
    st.write("You selected:", k)

    # Add a horizontal line for separation
    st.markdown("---")

    ###############################################
    # Step5: Clustering The Review & Review The Keyword For Each Cluster
    ###############################################
    st.header("Step5: Show the Key Word for Each Cluster")
    st.markdown("This step groups reviews into clusters and shows the top keywords in each group. Click the button below to see what each group is talking about.")
    if st.session_state.Page2_Step4:
        if st.button("Clustering Review & Show Key Word"):
            # Retrieve the dataframe from session state
            df = st.session_state.get('df', None)

            # Load vectorizer
            TFvectorizer = joblib.load("TFvectorizer.pkl")

            # Transfer the player review in to numeric form
            player_review_overall_vectorize = TFvectorizer.fit_transform(df['review_preprocessed'])
            model_overall = KMeans(n_clusters=k, init='k-means++', max_iter=1000, n_init=20)
            model_overall.fit(player_review_overall_vectorize)

            # Define a new cluster variable to store the number of clusters for each review
            df['cluster'] = model_overall.labels_

            # Check Keyword For Overall Clustering Model
            st.subheader("Top Keywords in Each Cluster")
            Cluster = model_overall.cluster_centers_.argsort()[:, ::-1]
            Gain_Keywords = TFvectorizer.get_feature_names_out()  # Set a function to gain keyword from cluster

            for i in range(k):
                with st.expander(f"Cluster {i} - Top Keywords"):
                    for keyword in Cluster[i, :10]:
                        st.write(Gain_Keywords[keyword])
            st.markdown('---')

            # Store the preprocessed dataframe in session state
            st.session_state.df = df

            # Step 5 Completed
            st.session_state.Page2_Step5 = True 
    else:
        st.warning("Run Step 4 First")

    ###############################################
    # Step6: Write Down the Topic For Each Cluster
    ###############################################
    st.header("Step6: Write Topic for Each Cluster")
    st.markdown("This step is to input your preferred topic name for each cluster.")
    if st.session_state.Page2_Step5:
        # Retrieve the dataframe from session state
        df = st.session_state.get('df', None)

        # Now, create a form for the user to input topics for each cluster at once
        with st.form(key="cluster_topics_form"):
            st.subheader("Enter a Topic for Each Cluster")

            # Define cluster_topics to store topic name from user
            cluster_topics = {}
            
            # Allow user to input topics for each cluster
            for i in range(k):
                topic = st.text_input(f"Topic for Cluster {i}:", key=f"topic_{i}")
                cluster_topics[i] = topic
            
            # Submit button
            submit_button = st.form_submit_button(label='Submit Topics')

        if submit_button:
            # Display the user-entered topics after submission
            st.subheader("Cluster Topics:")
            for i, topic in cluster_topics.items():
                if topic:
                    st.write(f"Cluster {i}: {topic}")
                else:
                    st.warning(f"Cluster {i} has no topic assigned.")


        # Create new 'topic' column based on cluster_topics dictionary
        df['Topic'] = df['cluster'].map(cluster_topics)

        # Show df dataframe
        st.dataframe(df)     

        # Shows Distribution of Topic
        st.subheader("Distribution of Topics (Clusters)")
        st.write(df['cluster'].value_counts())
        
        # Store the preprocessed dataframe in session state
        st.session_state.df = df

        # Step 6 Completed
        st.session_state.Page2_Step6 = True 
    else:
        st.warning("Run Step 5 First")

    ###############################################
    # Step7: Train Logistic Regression Model
    ###############################################
    st.header("Step7: Train Logistic Regression Model")
    st.markdown("This step is to start train the logistic regression model.")
    # Add checkbox for spelling correction
    do_random_under_sampling = st.checkbox("Apply random under-sampling if the topic distribution is imbalanced.")
    if st.session_state.Page2_Step6:
        if st.button("Train Model"):
            # Retrieve the dataframe from session state
            df = st.session_state.get('df', None)

            texts = df['review_preprocessed'].astype(str).tolist()
            labels = df['cluster'].tolist()
            y = np.array(labels)

            # Load vectorizer
            TFvectorizer = joblib.load("TFvectorizer.pkl")

            # Load vectorizer
            X_tfidf = TFvectorizer.transform(texts)

            # Data Splitting
            x_train, x_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42, stratify=y)

            # Under Random Sampling
            if do_random_under_sampling:
                x_train, y_train = Set_Up_Function.random_under_sampling(x_train, y_train)

                # Shows Distribution of Topic
                st.subheader("Y Train Distribution of Topics (Clusters)")

                # Convert y_train to Series to use value_counts()
                y_train_series = pd.Series(y_train)     
                st.write(y_train_series.value_counts())

                st.write("✅ Random Under Sampling.")
            else:
                st.write("❌ Skipped Random Under Sampling.")

            # Define the Logistic Regression model with the best hyperparameters
            lr_model = LogisticRegression(max_iter=1000)

            # Fit the model on the training data
            lr_model.fit(x_train, y_train)

            # Call Evaluation function
            Set_Up_Function.plot_learning_curve(LogisticRegression(max_iter=1000), x_train, y_train)
            Set_Up_Function.logistic_classification_report(lr_model, x_test, y_test)
            Set_Up_Function.plot_logistic_confusion_matrix(lr_model, x_test, y_test)

            # Save the model to session state
            st.session_state['lr_model'] = lr_model

            # Step 7 Completed
            st.session_state.Page2_Step7 = True 
    else:
        st.warning("Run Step 6 First")

    ###############################################
    # Step8: Download Trained Logistic Regression Model
    ###############################################
    st.header("Step8: Download Trained Logistic Regression Model")
    filename = st.text_input("Enter filename to save model:", value="Customize_model.pkl")
    st.markdown("This step is to dowload the trained model. So that you can be use for classify your review")
    st.markdown("⚠️ **Note:** If a file with the same name already exists in the folder, it will be overwritten.")
    if st.session_state.Page2_Step7:
        if st.button("📥 Save Model"):
                if 'lr_model' in st.session_state:
                    joblib.dump(st.session_state['lr_model'], filename)
                    st.success(f"✅ Model saved as {filename} & remember choose the file path.")

                    # Open for download
                    with open(filename, "rb") as f:
                        st.download_button("Choose File Path", f, file_name=filename)

                    # Step 7 Completed
                    st.session_state.Page2_Step8 = True 
                else:
                    st.error("❌ Logistic Regression model not found in session. Please train the model first.")
    else:
        st.warning("⚠️ Run Step 7 First")
        
    # Add a horizontal line for separation
    st.markdown("---")

    ###############################################
    # Download Classified Dataset Section (Optional)
    ###############################################
    st.header("Download Classified Dataset (Optional)")
    st.markdown("This step is to dowload the classified dataset.")
    # Input File Name
    class_file_name  = st.text_input("Enter File Name", value="Classified_Reviews.csv")
    st.write("You entered file name:", class_file_name )
    # Show download button directly if Step3 is completed
    if st.session_state.Page2_Step8:
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
        st.warning("Run Step 8 First")

    # Add a horizontal line for separation
    st.markdown("---")