import streamlit as st
import pandas as pd
import joblib
import spacy 
import requests
from streamlit_lottie import st_lottie
from io import BytesIO
#import sklearn
#pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.0/en_core_web_sm-2.2.0.tar.gz
#import en_core_web_sm
nlp = spacy.load("en_core_web_sm")

def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code !=200:
        return None
    return r.json()

lottie_morty = load_lottieurl("https://assets3.lottiefiles.com/packages/lf20_4jsnlwpe.json")

# model import from the one we trained in the pynb
# sarcasm_pickle = open(r'D:\random_projets\sarcasm_nlp\sarcasm_model.pkl', 'rb')
# sarcasm_model = joblib.load(sarcasm_pickle)

URI = 'https://github.com/LarryJr64/sarcasm_detection_NLP/blob/main/sarcasm_model.pkl?raw=true'
sarcasm_pickle = BytesIO(requests.get(URI).content)
sarcasm_model = joblib.load(sarcasm_pickle)


def text_prepro(texts):
    texts_clean = texts.replace('#','').replace(r'\s+', ' ').replace("'s", '').replace("s", '').replace("u", '')

    clean_container = []


    for text in nlp.pipe(texts_clean, disable=["tagger", "parser", "ner"]):

        txt = [token.lemma_.lower() for token in text 
               if token.is_alpha 
               and not token.is_stop 
               and not token.is_punct]

    clean_container.append(" ".join(txt))
  
    return clean_container

def main():
    st.title("Sarcasm detector")
    texte = st.text_input("Enter your sentence for sarcasm verification :")
    if texte:
        result = sarcasm_model.predict(text_prepro(pd.Series(texte)))
        if result ==[0]:
            st.write("This sentence is not sarcastic")
        else :
            st.write("This sentence is sarcastic")

                 
expander = st.sidebar.expander("What is this?")
expander.write(
    """
this application try to classify your sentence if it is sarcastic or not. Please only type in english.
"""
)


if __name__ == '__main__':
    main()
    with st.container() :
        st.write("---")
        left_column, right_column = st.columns(2)
        with left_column :
            st.write("##")
            st.write(
                """
                Your face when people are sarcastic around you 
                but you dont get it           ==> 
                
                But don't worry !! 
                
                Now with your splendid sarcasm detector you can
                understand their sarcasm and turn yourself 
                into evil morty
                """)
        with right_column :
            st_lottie(lottie_morty, height=250, key="morty")
