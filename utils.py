#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import nltk, re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

import string
import pandas as pd

from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='darkgrid', font_scale=2)
import numpy as np
import plotly.graph_objects as go

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

# spacy for lemmatization
import spacy

# Plotting tools
import pyLDAvis
import pyLDAvis.gensim

import gc
import warnings
warnings.filterwarnings("ignore")


# In[ ]:


STOP_WORDS = stopwords.words('english')


# In[ ]:


def clean(text):
    
    text = ''.join([word for word in text if word not in string.punctuation]) # без пунктуаций
    text = text.lower() # нижний регистр
  
    return text


def word_tokenize(text):

    lemmatizer = WordNetLemmatizer()
    text = clean(text)
    tokens = [token for token in nltk.word_tokenize(text) if token not in STOP_WORDS]
    tokens = [token for token in tokens if re.match(r"^\d*[a-z][\-.0-9:_a-z]{1,}$", token)]
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    return tokens


def sent_tokenize(text):
     
    tokens = [token for token in nltk.sent_tokenize(text) if len(token) > 5]
    
    return tokens


# Word Cloud colors
def wcolors(word=None, font_size=None, position=None,  orientation=None, font_path=None, random_state=None):
    colors = ["#7e57c2", "#03a9f4", "#011ffd", "#ff9800", "#ff2079"]
    return np.random.choice(colors)


# Word Cloud visualization
def wordcloud(df, title = None):
    # Set random seed to have reproducible results
    np.random.seed(64)
    
    wc = WordCloud(
        background_color="white",
        max_words=200,
        max_font_size=40,
        scale=5,
        random_state=0
    ).generate_from_frequencies(df)

    wc.recolor(color_func=wcolors)
    
    fig = plt.figure(1, figsize=(15,15))
    plt.axis('off')

    if title:
        fig.suptitle(title, fontsize=14)
        fig.subplots_adjust(top=2.3)

    plt.imshow(wc),
    plt.show()
    
    
''' СТОП-СЛОВА БИГРАММЫ/ТРИГРАММЫ '''

def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in STOP_WORDS] for doc in texts]

# def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
#     """https://spacy.io/api/annotation"""
#     texts_out = []
#     for sent in texts:
#         doc = nlp(" ".join(sent)) 
#         texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
#     return texts_out

def lemmatize(tokens):
    
    ''' СОЗДАНИЕ БИ- И ТРИГРАМ И ДОБАВЛЕНИЕ ИХ В СПИСОК ТОКЕНОВ СТАТЕЙ  '''

    '''

    min_count - ignore all words and bigrams with total collected 
                count lower than this. Bydefault it value is 5
    
    threshold - represents a threshold for forming the phrases 
                (higher means fewer phrases). A phrase of words a 
                and b is accepted if 
                (cnt(a, b) - min_count) * N / (cnt(a) * cnt(b)) > threshold, 
                where N is the 
                total vocabulary size. Bydefault it value is 10.0
    '''
       
    bigram = gensim.models.Phrases(tokens, min_count=5, threshold=100)
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    
    ''' УДАЛЯЕМ СТОПЫ '''
    data_words_nostops = remove_stopwords(tokens)

    ''' СОЗДАЁМ БИГРАММЫ '''
    data_words_bigrams = [bigram_mod[doc] for doc in data_words_nostops]
    
    ''' ЛЕММАТИЗАЦИЯ оставляя только noun, adj, vb, adv'''

    lemmatizer = WordNetLemmatizer()
    
    data_lemmatized = []
        
    for i in range(len(data_words_bigrams)): 
        sentence = []
        
        for j in range(len(data_words_bigrams[i])):
                sentence.append(lemmatizer.lemmatize(data_words_nostops[i][j]))
            
        data_lemmatized.append(sentence)
        
    return data_lemmatized


def dominant_topic(ldamodel, corpus, sources, filenames):
    sent_topics_df = pd.DataFrame() 

    for i, row in enumerate(ldamodel[corpus]):
        row = sorted(row[0], key=lambda x: (x[1]), reverse=True)
        
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0: 
                
                ''' ВЕКТОР topn-СЛОВ ТЕМЫ topic_num вместе с массой '''
                wp = ldamodel.show_topic(topic_num, topn = 10)

                ''' ВЫДЕЛЯЕМ topn-КЛЮЧЕВЫЕ СЛОВА '''
                topic_keywords = ', '.join([word for word, prop in wp])

                sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num),                                                                   round(prop_topic,4),                                                                   topic_keywords]),                                                        ignore_index=True)
            else:
                break
         
    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']
    sent_topics_df['Article'] = pd.Series(sent_topics_df.index).apply(lambda x: sources.loc[sources.label 
                                                                                            == int(filenames[x]),
                                                                                            'name'].values[0])
    
    sources.loc[sources.label == int(filenames[i]), 'name'].values[0]

    return(sent_topics_df)


# In[ ]:




