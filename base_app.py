import streamlit as st
import joblib
import os
import pandas as pd
from PIL import Image

# File paths for models and images
base_dir = os.path.dirname(__file__)
paths = {
    "tfidf_vectorizer": os.path.join(base_dir, 'models', 'tfidf_vectorizer.pkl'),
    "lr_model": os.path.join(base_dir, 'models', 'lr_classifier_model.pkl'),
    "nb_model": os.path.join(base_dir, 'models', 'nb_classifier_model.pkl'),
    "rf_model": os.path.join(base_dir, 'models', 'rf_classifier_model.pkl'),
    "svm_model": os.path.join(base_dir, 'models', 'svm_classifier_model.pkl'),
    "data": os.path.join(base_dir, 'data', 'train.csv'),
    "logo_image": os.path.join(base_dir, 'images', 'Logo1.jpg'),
    "new_word_cloud": os.path.join(base_dir, 'images', 'New word cloud.png'),
    "balanced_class_dist": os.path.join(base_dir, 'images', 'Balanced Class distribution.png'),
    "class_dist": os.path.join(base_dir, 'images', 'Class distribution.png'),
    "announcement_image": os.path.join(base_dir, 'images', 'announcement-article-articles-copy-coverage.jpg'),
    "f1_scores": os.path.join(base_dir, 'images', 'F1 Scores.png'),
    "model_evaluation": os.path.join(base_dir, 'images', 'Model Evaluation.png'),
    "correlation_matrix": os.path.join(base_dir, 'images', 'Correlation Matrix.png'),
    "training_time": os.path.join(base_dir, 'images', 'Training time.png')
}

# Loading models and vectorizer
vectorizer = joblib.load(paths["tfidf_vectorizer"])
models = {
    "Logistic Regression": joblib.load(paths["lr_model"]),
    "Naive Bayes": joblib.load(paths["nb_model"]),
    "Random Forest": joblib.load(paths["rf_model"]),
    "Support Vector Machine": joblib.load(paths["svm_model"]),
}

def main():
    """Streamlit News Classification App"""

    st.title("News Classifier Application")
    st.subheader("A Comprehensive Tool for News Article Categorization")

    st.sidebar.markdown(
        """
        <div style="background-color:#28a745;padding:10px;border-radius:10px;">
            <p style="color:white;"><b>Navigation Tip:</b></p>
            <p style="color:white;">Use the dropdown menu to switch between sections.</p>
        </div>
        """, unsafe_allow_html=True)

    pages = ["Home", "Information", "EDA", "Prediction", "Feedback", "About Us"]
    choice = st.sidebar.selectbox("Navigate to", pages)

    if choice == "Home":
        st.info("Welcome to the News Classifier App!")
        st.markdown(
            """
            This application classifies news articles into various categories like sports, education, entertainment, business, and technology using machine
            learning models.
            Use the sidebar to explore different sections including Information, EDA, Prediction, Feedback, and About Us.
            """
        )
        st.image(Image.open(paths["announcement_image"]), caption='Welcome to the News Classifier App!')

    elif choice == "Information":
        st.info("General Information")
        st.markdown(
            """
            This tool classifies news articles into categories: sports, education, entertainment, business, and technology.
            
            ### Models Utilized:
            - **Multinomial Naive Bayes:** Probabilistic text classification model.
            - **Random Forest:** Ensemble learning method for enhanced accuracy.
            - **Logistic Regression:** Statistical model for binary/multi-class classification.
            - **Support Vector Machine (SVM):** Model for finding the optimal hyperplane in classification.
            """
        )

    elif choice == "EDA":
        st.info("Exploratory Data Analysis")
        st.markdown(
            """
            ### Data Overview:
            - **Distribution Analysis:** 
              - **Category Distribution:** Explore the distribution of news articles across different categories to understand the prevalence and imbalance of
              categories in the dataset.
              - **Word Frequency Analysis:** Identify the most common words and phrases used in each category, which helps in understanding the content and
              themes prevalent in different types of news.
              - **Length Analysis:** Examine the length of articles in terms of word count and character count across different categories to identify any
              notable differences.

            - **Labeling and Dataset Characteristics:** 
              - **Labeled Dataset:** The dataset consists of news articles labeled into categories such as sports, education, entertainment, business, and
              technology. These labels are essential for training supervised machine learning models.
              - **Source and Timeframe:** Information about the source of the dataset and the timeframe over which the articles were collected can provide
              context for the analysis.

            ... (other sections omitted for brevity)
            """
        )
        
        st.subheader("Imbalanced Class Distribution")
        st.image(Image.open(paths["class_dist"]), caption='Distribution of Articles by Category')
        
        if st.button("View Balanced Data"):
            st.image(Image.open(paths["balanced_class_dist"]), caption='Balanced Distribution of Articles by Category')

        st.subheader("Word Cloud by Category")
        st.image(Image.open(paths["new_word_cloud"]), caption='Most Used Words in Each Category')

        ... (other sections omitted for brevity)

    elif choice == "Prediction":
        st.info("Make Predictions")
        model_choice = st.sidebar.radio("Select Model", list(models.keys()))

        st.session_state.selected_model = model_choice
        selected_model = models[model_choice]

        news_text = st.text_area("Enter news text for classification", "")
        if st.button("Classify"):
            vect_text = vectorizer.transform([news_text])
            prediction = selected_model.predict(vect_text)[0]
            st.success(f"The article is categorized as: {prediction}")

    elif choice == "Feedback":
        st.info("Feedback")
        st.markdown(
            """
            ### We Value Your Feedback
            Your feedback is essential for us to improve this application. Please share your comments, suggestions, and any issues you've encountered. Whether
            it's about the user interface, functionality, or any new features you'd like to see, we are eager to hear from you.
            
            #### What You Can Provide Feedback On:
            - **User Experience:** How easy and intuitive is it to navigate the app? Are there any difficulties or confusions?
            - **Performance:** Is the app running smoothly? Have you experienced any lags or crashes?
            - **Features:** Are there any features you love or any you think are missing? How can we enhance the current features?
            - **Accuracy:** Are the predictions accurate? How well do the models categorize the news articles?

            We appreciate your feedback and are committed to making this app better for you.

            **Your Feedback:**
            """
        )
        feedback = st.text_area("Your Feedback", "")
        if st.button("Submit Feedback"):
            st.success("Thank you for your feedback! We will use it to improve the application.")

    elif choice == "About Us":
        st.info("About Us")
        st.image(Image.open(paths["logo_image"]), width=300)
        st.markdown(
            """
            ## About Us

            This application was developed by Team EG-3 from ExploreAI Academy. It leverages machine learning models to classify news articles into various categories such as sports, education, entertainment, business, and technology.

            ### New Mission
            In today's fast-paced digital landscape, the ability to effectively categorize and deliver news content is crucial for enhancing user experience and operational efficiency in news outlets. As data science consultants for a prominent news organization, our team has undertaken the task of...
            """
        )

if __name__ == '__main__':
    main()
