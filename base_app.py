# Importing Streamlit and other essential libraries
import streamlit as st
import joblib
import os
import pandas as pd
from PIL import Image

# File paths for models and vectorizer
base_dir = os.path.dirname(__file__)
paths = {
    "tfidf_vectorizer": os.path.join(base_dir, 'models', 'tfidf_vectorizer.pkl'),
    "lr_model": os.path.join(base_dir, 'models', 'lr_classifier_model.pkl'),
    "nb_model": os.path.join(base_dir, 'models', 'nb_classifier_model.pkl'),
    "rf_model": os.path.join(base_dir, 'models', 'rf_classifier_model.pkl'),
    "svm_model": os.path.join(base_dir, 'models', 'svm_classifier_model.pkl'),
    "data": os.path.join(base_dir, 'data', 'train.csv'),
    "logo_image": os.path.join(base_dir, 'images', 'Logo1.jpg'),
    "new_wordcloud": os.path.join(base_dir, 'images', 'New word cloud.png'),
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

            - **Exploratory Data Analysis (EDA):**
              - **Pattern Identification:** Use EDA to uncover patterns, trends, and anomalies in the data that can inform feature engineering and model
              selection.
              - **Bias Detection:** Detect and analyze biases in the dataset, such as over-representation or under-representation of certain categories, which
              can impact model performance.
              - **Correlation Analysis:** Investigate correlations between different features (e.g., publication date, article length) and the target
              categories.
              - **Sentiment Analysis:** Perform sentiment analysis to understand the emotional tone of articles across different categories and how it may
              affect classification.
              - **Visualization:** Use visualizations such as histograms, bar charts, word clouds, and box plots to present the data intuitively and
              insightfully.
              - **Missing Values:** Check for missing values and understand their distribution and impact on the dataset.

            - **Model Development:**
              - **Feature Selection:** EDA helps in selecting the most relevant features for model development by identifying important patterns and
              correlations.
              - **Data Cleaning:** Identify and address any data quality issues such as duplicates, inconsistencies, or errors in the dataset.
              - **Balancing the Dataset:** Techniques such as oversampling, undersampling, or using synthetic data generation (e.g., SMOTE) to balance the
              dataset and improve model performance.

            - **Model Evaluation:**
              - **F1 Scores:** Evaluate the performance of different models using F1 scores, which provide a balance between precision and recall. F1 scores
              are particularly useful for imbalanced datasets.
              - **Confusion Matrix:** Analyze the confusion matrix for each model to understand the distribution of true positives, false positives, true
              negatives, and false negatives.
              - **ROC and AUC:** Examine the Receiver Operating Characteristic (ROC) curve and Area Under the Curve (AUC) for each model to evaluate their
              discriminatory power.
              - **Precision-Recall Curve:** Investigate the precision-recall curve to assess the trade-off between precision and recall at different threshold
              levels.

            - **Correlation Matrix:**
              - **Feature Correlation:** Generate and analyze the correlation matrix to understand the relationships between different features. This helps in
              identifying multicollinearity and selecting features that contribute the most to the model's performance.

            - **Training Time:**
              - **Model Training Duration:** Measure and compare the training time for different models to understand their computational efficiency and
              scalability.
              - **Optimization:** Explore optimization techniques to reduce training time without compromising model accuracy, such as using faster
              algorithms, reducing data dimensionality, or parallelizing computations.
            """
        )
        
        st.subheader("Imbalanced Class Distribution")
        st.image(Image.open(paths["class_dist"]), caption='Distribution of Articles by Category')
        
        if st.button("View Balanced Data"):
            st.image(Image.open(paths["balanced_class_dist"]), caption='Balanced Distribution of Articles by Category')

        st.subheader("Word Cloud by Category")
        st.image(Image.open(paths["new_wordcloud"]), caption='Most Used Words in Each Category')

        if st.button("View F1 Scores"):
            st.image(Image.open(paths["f1_scores"]), caption='F1 Scores for Different Models')

        if st.button("View Model Evaluation"):
            st.image(Image.open(paths["model_evaluation"]), caption='Model Evaluation Metrics')

        if st.button("View Correlation Matrix"):
            st.image(Image.open(paths["correlation_matrix"]), caption='Correlation Matrix of Features')

        if st.button("View Training Time"):
            st.image(Image.open(paths["training_time"]), caption='Training Time for Different Models')

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
            In today's fast-paced digital landscape, the ability to effectively categorize and deliver news content is crucial for enhancing user experience and operational efficiency in news outlets. As data science consultants for a prominent news organization, our team has undertaken the task of developing robust classification models using advanced machine learning techniques. This project aims to demonstrate the application of natural language processing (NLP) methodologies through an end-to-end workflow, leveraging Python and Streamlit for model deployment.

            ### Project Overview
            The News Classifier Application is a comprehensive tool designed to provide accurate categorization of news articles. Our goal is to make it easier for users to quickly identify the category of any given news article, enhancing the overall news consumption experience.

            ### Contact Information
            For more information or inquiries, feel free to contact us at: **eg3_classification@sandtech.co.za**

            ### Supervisors
            - **Marc Marais**: [Mmarais@sandtech.com](mailto:Mmarais@sandtech.com)
            - **Oladare Adekunle**: [Oadekunle@sandtech.com](mailto:Oadekunle@sandtech.com)
            - **Ereshia Gabier**: [Egabier@sandtech.com](mailto:Egabier@sandtech.com)

            ### Team Members
            - **Mieke Spaans**: Project Manager
            - **Sinawo Londa**: Team Lead
            - **Pfarelo Ramunasi**: GitHub Manager
            - **Coceka Keto**: Data Scientist
            - **Simphiwe Khoza**: Data Scientist
            - **Zakhele Mabuza**: Data Scientist

            ### Team Roles and Contributions
            - **Mieke Spaans**: Ensured the project was on track and met deadlines, coordinated team activities, and managed communications.
            - **Sinawo Londa**: Led the team, provided guidance, and ensured the successful execution of project goals.
            - **Pfarelo Ramunasi**: Managed the GitHub repository, ensured version control, and handled code integration.
            - **Coceka Keto**: Focused on data cleaning, preprocessing, and initial exploratory data analysis.
            - **Simphiwe Khoza**: Developed machine learning models and evaluated their performance.
            - **Zakhele Mabuza**: Worked on the deployment of the application, created visualizations, and handled backend integration.

            ### Acknowledgments
            We extend our gratitude to our supervisors for their invaluable guidance and support throughout this project. Special thanks to ExploreAI Academy
            for providing us with this opportunity to apply our skills in a real-world scenario.

            ### Conclusion
            We achieved our goal and created a transformative tool designed to enhance the way you interact with the world of news. It prioritizes your
            preferences and curates content that truly matters to you. We are redefining the news consumption experience.

            Our commitment to personalization, innovation, and user satisfaction ensures that you stay informed in a way that is both meaningful and enjoyable.

            **Join Us** in our mission to bring innovative data solutions to the world. Follow our journey and contribute to the future of news classification
            technology.

            ### Additional Information
            For more details about our project and future updates, visit our GitHub repository or contact any of our team members.

            Together, we are committed to driving innovation and making a positive impact through data science and technology.
            """
        )
        
if __name__ == "__main__":
    main()

