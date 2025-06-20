# SENTIMENT-ANALYSIS

*COMPANY* :CODTECH IT SOLUTIONS

*NAME* :PULLA HARI SAIRAM REDDY

*INTERN ID* :CT06DF1059

*DOMAIN* :DATA ANALYTICS

*DURATION* :6 WEEKS

*MENTOR* :NEELA SANTHOSH KUMAR

**Airline Tweet Sentiment Analyzer**

 **Objective**

The goal of this project is to build a sentiment analysis system using Natural Language Processing (NLP) techniques. The system should be able to identify whether a piece of text expresses a positive, negative, or neutral sentiment. In this case, we are working with real tweets posted by people about airline services. By processing and analyzing this data, the project helps understand how customers feel about different airlines based on their tweets.

This task was given as part of an internship project, where the main deliverable was to showcase **data preprocessing**, **model implementation**, and **insights** using a Jupyter Notebook. The project uses standard NLP and machine learning tools and follows an industry-level workflow for text classification.

**Dataset Details**

The dataset used for this project is the **Twitter US Airline Sentiment Dataset**, which is available publicly on Kaggle. It contains more than 14,000 rows of tweets directed at major U.S. airlines like United, Delta, American, Southwest, and more.

Each row includes:

* The **tweet text** written by the user.
* The **sentiment label** (positive, negative, or neutral).
* Other metadata like the airline name, date/time, and user information (not used in this analysis).

For simplicity and focus, we use only the tweet text and the sentiment label to build the model.

**Tools & Libraries Used**

This project was implemented using the following tools and Python libraries:

* **Python** – Core programming language.
* **Jupyter Notebook / Google Colab** – Development environment.
* **Pandas** – For reading and managing datasets.
* **Matplotlib & Seaborn** – For visualizing distributions and results.
* **NLTK (Natural Language Toolkit)** – For text cleaning, stopword removal, and stemming.
* **Scikit-learn** – For TF-IDF vectorization, model training, and evaluation.

These tools are widely used in the data science community and are perfect for building NLP pipelines from scratch.

 **Implementation Workflow**

1. **Data Loading**: The dataset was loaded and cleaned using pandas.
2. **Exploratory Data Analysis (EDA)**: The distribution of sentiments was visualized, showing that most tweets were negative.
3. **Text Preprocessing**: All tweets were cleaned by:

   * Removing URLs, mentions, hashtags, numbers, and punctuation
   * Converting text to lowercase
   * Removing stopwords
   * Applying stemming
4. **Feature Engineering**: TF-IDF (Term Frequency–Inverse Document Frequency) vectorization was used to convert text data into numerical format.
5. **Model Training**: A Multinomial Naive Bayes classifier was trained on the processed text.
6. **Model Evaluation**: The model’s performance was checked using accuracy, a classification report, and a confusion matrix.
   
 **Insights Gained**

* **Negative tweets** were the most common, indicating that customers often use Twitter to report problems.
* The **Naive Bayes model** performed well for a basic sentiment classification task.
* The model was most accurate in identifying **negative and positive tweets**, while some confusion occurred with **neutral tweets**, which is expected in real-world data.
* Preprocessing played a **huge role** in improving the quality of predictions.

**Future Enhancements**

While the current model works effectively, several improvements can be made:

* Use advanced models like **Logistic Regression**, **Support Vector Machines**, or even **deep learning (BERT)** for higher accuracy.
* Add **emoji and hashtag interpretation** to capture emotions more precisely.
* Include **airline name** as a feature to see which airlines get more negative/positive feedback.
* Create a **dashboard** to visualize live sentiment trends if connected to the Twitter API.
* 
**Conclusion**

This project successfully demonstrates how to perform sentiment analysis on real-world textual data using simple but effective machine learning techniques. It walks through the complete process: from cleaning raw tweets to building a predictive model. The use of basic models like Naive Bayes helps in understanding the fundamentals, and the project lays the foundation for more complex text analysis in the future.


