import pandas as pd
import spacy
from sklearn.feature_extraction.text import CountVectorizer
import streamlit as st 
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB



st.title("NEWS CATEGORY CLASSIFICATION")
df=pd.read_json("news_dataset.json")
dic={'category':{"SCIENCE": 1, "BUSINESS": 2, "CRIME": 3, "SPORTS": 4}}
df=df.replace(dic)
step=[("VECTORIZATION",CountVectorizer(ngram_range=(1,2))),("CLASSIFIER",MultinomialNB())]
pipe=Pipeline(step)
pipe.fit(df.text,df.category)
input=st.text_area("GIVE NEWS FOR CLASSIFICATION")
cat=pipe.predict([input])
if input:
   if cat[0]==1:
       st.success("SCIENCE RELATED NEWS")
   elif cat[0]==2:
      st.success("BUSSINESS RELATED NEWS")
   elif cat[0]==3:
        st.success("CRIME RELATED NEWS")
   elif cat[0]==4:
         st.success("SPORTS RELATED NEWS")

