import streamlit as st
import requests
from PIL import Image



ORL = 'https://github.com/LarryJr64/sarcasm_detection_NLP/blob/main/mots-nlp.png?raw=true'
response = requests.get(ORL)
with open("image.jpg", "wb") as f:
    f.write(response.content)
image = Image.open("image.jpg")

def main() : 
    st.title("Sarcasm Detector")
    st.write('##')

    st.subheader("Why this project")
    st.write(
        """
        The goal of this prject was to create a tool which analyze sentences and detect
        if the sentences is sarcastic or not using NLP (Natural Language Processing).  
        
        First of all, before any model creation we need data. The database come from Kaggle.
        This dataset is filled with articles headlines of two different websites. huffingpost and theonion
        This dataset contains two different labels, 1 for sarcastic and 0 if not sarcastic.
        TheOnion is known to publish very sarcastic news helping us with high quality sarcastic healines.
        """
        )
    st.write(
        """
        [here find the dataset](https://www.kaggle.com/datasets/saurabhbagchi/sarcasm-detection-through-nlp)
        """
        )
    st.write("##")
    st.subheader("How it works")   
    st.write(
        """
        We had more or less 50/50 with sarcastic sentences and not sarcastics headlines for a total of 26.000 headlines.
        We then cleaned the database, removed stop_words, etc... and convert all words to tokens. After getting our bag of words 
        we vectorized it and use a logistic regression to train a model. 
        
        We end up having a 90% precision with this model.
        """
        )

    st.write("##")
    st.subheader("Limits") 
    with st.container() :
        left_column, right_column = st.columns(2)
        with left_column :
            st.write("##")
            st.write(
                """
                Our model has 2 major limits  
                
                FIRST :
                The model only works with english sentences.
                If you try wwith an other languages the model
                prediction is totally random.
                    
                SECOND :
                As you can see the word having the 
                most impact are a bit weird. 
                Take nation for example, 
                as we say TheOnion is using 
                sarcasm to critic most of famous 
                people and politics. 
                This is a reason why
                some words are tops, it is a 
                politics sarcasm oriented. 
                In other words our sarcasm 
                detector is more likely a politic
                sarcasm detector.
                """)
        with right_column :
            st.image(image, width=160)
    
if __name__ == '__main__' :
    main()