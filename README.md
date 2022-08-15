# Project Name - Sentiment Analysis of Restuarant Reviews

* Applied Natural Language Processing techniques on the dataset containing reviews.
     *      Pre-processing using regex; tokenization
     *      Lemmatization
     *      TF-IDF Vectorizer
* Trained the dataset using two classification algorithms (Multinomial Naive Bayes,Logistic Regression) and optimized each using GridSearchCV. 

## Data Cleaning
After scraping the data, I needed to clean it up so that it was usable for our model. I made the following changes and created the following variables:

*	Parsed numeric data out of salary 
*	Made columns for employer provided salary and hourly wages 
*	Removed rows without salary 
*	Parsed rating out of company text 
*	Made a new column for company state 
*	Added a column for if the job was at the company’s headquarters 
*	Transformed founded date into age of company 
*	Made columns for if different skills were listed in the job description:
    * Python  
    * R  
    * Excel  
    * AWS  
    * Spark 
*	Column for simplified job title and Seniority 
*	Column for description length 
