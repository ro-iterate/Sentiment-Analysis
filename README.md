# Sentiment-Analysis 

Python file: run in IDE with Python3 

Sentiment analysis of Amazon product reviews using Natural Language Processing (NLP). Dataset available at <a href = https://www.kaggle.com/datasets/datafiniti/consumer-reviews-of-amazon-products>Reviews of Amazon Products</a>. 

The reviews were isolated from the rest of the data set. A random review is chosen, preprocessed (including lemmatisation, removal of stop words) and its sentiment determined using spaCy and compared to the review rating. 
The review is then compared to another randomly chosen, preprocessed review using the TextBlob library.  
