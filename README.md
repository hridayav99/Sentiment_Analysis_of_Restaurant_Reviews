# Project Name - Sentiment Analysis of Restuarant Reviews

*	Applied Natural Language Processing techniques on the dataset containing reviews:
    * Pre-processing using regex; tokenization
    * Lemmatization
    * TF-IDF Vectorizer     
*	Trained the dataset with two classification algorithms (Multinomial Naive Bayes,Logistic Regression) and optimized both using GridSearchCV. 

## Dataset Description

The dataset consists of 1000 entries. There are two columns in the dataset- review and liked. The liked column is the target variable. It consists of two classes- 0 for negative sentiment and 1 for positive sentiment. Both the classes are represented equally.

## Data Cleaning

First, I split the dataset into two sets- train(80%) and test(20%) sets to avoid data leakage. Applied the following steps on each set:
*  Removed whitespaces, punctuations.
*  Changed text to lower case.
*  Tokenized the text and removed stopwords.
*  Applied lemmatization.

## Model Building

After data cleaning, I transformed the raw text into matrix form using TfIdf vectorizer.

I used 2 classification algorithms-

*  Multinomial Naive Bayes
*  Logistic Regression

I evaluated both the models using accuracy score and optimized them using GridSearchCV with 5 folds.

## Model Evaluation

| Syntax      | Description | Test Text     |
| :---        |    :----:   |          ---: |
| Header      | Title       | Here's this   |
| Paragraph   | Text        | And more      |
