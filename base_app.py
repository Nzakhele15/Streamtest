# Importing Streamlit and other essential libraries
import streamlit as st
import joblib
import os
import pandas as pd
from PIL import Image

# File paths for models and vectorizer
base_dir = r"C:\Users\Zakhele\Downloads\Compressed\StreamlitApp\Main2"
paths = {
    "tfidf_vectorizer": os.path.join(base_dir, 'tfidf_vectorizer.pkl'),
    "lr_model": os.path.join(base_dir, 'lr_classifier_model.pkl'),
    "nb_model": os.path.join(base_dir, 'nb_classifier_model.pkl'),
    "rf_model": os.path.join(base_dir, 'rf_classifier_model.pkl'),
    "svm_model": os.path.join(base_dir, 'svm_classifier_model.pkl'),
    "data": os.path.join(base_dir, 'train.csv'),
    "wordcloud": os.path.join(base_dir, 'wordcloud_by_category.png'),
    "imbalanced_dist": os.path.join(base_dir, 'imbalanced_distribution.png'),
    "balanced_class_dist": os.path.join(base_dir, 'balanced_class_category.png'),
    "stone_image": os.path.join(base_dir, 'stone.png'),
    "screenshot_image": os.path.join(base_dir, 'Screenshot_2024-07-04_203832.png')
}

# Function to check if files exist
def check_files(paths):
    missing_files = []
    for key, path in paths.items():
        if not os.path.exists(path):
            missing_files.append((key, path))
    return missing_files

# Check for missing files
missing_files = check_files(paths)
if missing_files:
    st.error("The following files are missing:")
    for key, path in missing_files:
        st.error(f"{key}: {path}")
else:
    # Loading vectorizer and models
    vectorizer = joblib.load(paths["tfidf_vectorizer"])
    models = {
        "Logistic Regression": joblib.load(paths["lr_model"]),
        "Naive Bayes": joblib.load(paths["nb_model"]),
        "Random Forest": joblib.load(paths["rf_model"]),
        "SVM": joblib.load(paths["svm_model"]),
    }

    # Load the training data
    data = pd.read_csv(paths["data"])

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
                This application classifies news articles into various categories like sports, education, entertainment, business, and technology using machine learning models.
                Use the sidebar to explore different sections including Information, EDA, Prediction, Feedback, and About Us.
                """
            )
            st.image(Image.open(paths["screenshot_image"]), caption='Welcome to the News Classifier App!')

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
                - Distribution analysis of news articles across different categories.
                - The dataset is labeled to train models for text classification.
                - EDA helps identify patterns and biases for accurate model development.
                """
            )
            st.subheader("Imbalanced Class Distribution")
            st.image(Image.open(paths["imbalanced_dist"]), caption='Distribution of Articles by Category')
            
            if st.button("View Balanced Data"):
                st.image(Image.open(paths["balanced_class_dist"]), caption='Balanced Distribution of Articles by Category')

            st.subheader("Word Cloud by Category")
            st.image(Image.open(paths["wordcloud"]), caption='Most Used Words in Each Category')

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
            st.markdown("We appreciate your feedback. Please share your comments and suggestions to help us improve this app.")
            feedback = st.text_area("Your Feedback", "")
            if st.button("Submit Feedback"):
                st.success("Thank you for your feedback!")

        elif choice == "About Us":
            st.info("About Us")
            st.image(Image.open(paths["stone_image"]), width=300)
            st.markdown(
                """
                This application was developed by Team EG-3 from ExploreAI Academy. It uses machine learning models to classify news articles.
                For more info, contact us at eg3_classification@sandtech.co.za.

                **Our Mission:**
                "Innovating Africa and the world through data-driven insights, one article at a time."
                """
            )

    if __name__ == '__main__':
        main()
