import pandas as pd
import json
import sentiments_algos as senti

honeywell_reviews = pd.read_excel('Dataset.xlsm', names=['id','answer'])
print(honeywell_reviews.head())

for index,row in honeywell_reviews.iterrows():
    senti_val, confidence = sentiment(row['answer'])
    
    print(row['answer']) 
    print(str(senti_val),str(confidence))