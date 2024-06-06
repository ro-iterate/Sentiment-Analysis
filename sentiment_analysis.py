# Import libraries 
import spacy 
from textblob import TextBlob 
import random 
import pandas as pd 


# Preprocess data 
def preprocess(text): 
    reviews = nlp(text)  
    
    # Lemmatise, lowercase, remove stop words and other characters  
    reviews = [token.lemma_.lower() for token in reviews if not token.is_stop
               and not token.is_punct and not token.is_currency and not token.is_digit] 
    reviews = ' '.join(reviews)
    
    return reviews
   

# Sentiment analysis     
def sentiment_analysis(text): 
    blob = TextBlob(text)
    
    # Polarity: closer to 1 = positive, -1 = negative 
    polarity = blob.sentiment.polarity 
    # Objectivity: 0 = very objective, 1 = very subjective 
    subjectivity_score = blob.sentiment.subjectivity 
    
    # Classify polarity score 
    if polarity > 0: 
        mood = 'Positive' 
    elif polarity < 0: 
        mood = 'Negative'  
    else: 
        mood = 'Neutral' 
    
    # Classify subjectivity score 
    if subjectivity_score > 0.5: 
        subjectivity = 'Subjective' 
    else: 
        subjectivity = 'Objective'
        
    return mood, polarity, subjectivity, subjectivity_score 
        

# Similarity analysis 
def similarity_analysis(text1, text2): 
    for token in text1: 
        token = nlp(token) 
        for token_ in text2: 
            token_ = nlp(token_) 
            similarity_score = token_.similarity(token) 
    return similarity_score

 
# Load model and read csv 
nlp = spacy.load('en_core_web_sm') 
dataframe = pd.read_csv('amazon_product_reviews.csv') 

print('Shape:',dataframe.shape, 'Info:', dataframe.info()) 


# Remove missing values 
clean_data = dataframe.dropna(subset = ['reviews.text'])  
cleaned_data = clean_data['reviews.text'] 

print('Shape cleaned data:',cleaned_data.shape)
   

# Sentiment analysis of sample review 
nr = [random.randint(0, 28332) for x in range(2)] 

sample = preprocess(cleaned_data[nr[0]]) 
senti_sample = sentiment_analysis(sample) 

print('\nSample Review', nr[0], ':', cleaned_data[nr[0]], 
      '\n\nSentiment Analysis:', senti_sample, '\nRating', dataframe['reviews.rating'][nr[0]]) 


# Similarity analysis between 2 reviews 
sample2 = preprocess(cleaned_data[nr[1]]) 
sim_sample = similarity_analysis(sample, sample2)  

print('\n\nSimilarity Analysis of Review', nr[0], 'and Review', nr[1], ':', sim_sample)
print('\n\nSample Review', nr[0], ':', cleaned_data[nr[0]], '\nSample Review', nr[1], 
      ':', cleaned_data[nr[1]])  