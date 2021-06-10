#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import nltk, re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

import string
import pandas as pd

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='darkgrid', font_scale=2)
import numpy as np
import plotly.graph_objects as go

import warnings
warnings.filterwarnings("ignore")

import spacy
from spacy.tokens import Doc
from spacy.pipeline import Sentencizer
from spacy.lang.en import English
import scispacy

from math import sqrt
from operator import itemgetter
import pytextrank


# In[ ]:


def limits(l):
    
    if l < 10000:
        SUMMARY_SENT_NUMBER = 5
        LIMIT_PHRASES = 4
        
    if (l >= 10000) & (l < 100000):     
        SUMMARY_SENT_NUMBER = 7
        LIMIT_PHRASES = 5
        
    if (l >= 100000) & (l < 500000):  
        SUMMARY_SENT_NUMBER = 8
        LIMIT_PHRASES = 6
        
    if l >= 500000:
        SUMMARY_SENT_NUMBER = 10
        LIMIT_PHRASES = 7
    
    return LIMIT_PHRASES, SUMMARY_SENT_NUMBER


# In[ ]:


def prepare_doc(text):

    nlp = spacy.load("en_core_sci_sm")
    
    tr = pytextrank.TextRank()
    nlp.add_pipe(tr.PipelineComponent, name="textrank", last=True)
    
    doc = nlp(text)
        
    return doc


# In[ ]:


def top_phrases(doc, LIMIT_PHRASES):
    
    print('\nКлючевые фразы:')
    sent_bounds = [[s.start, s.end, set([])] for s in doc.sents]

    phrase_id = 0
    unit_vector = []
    
    for p in doc._.phrases:
#         print("Phrase {0}: --{1}--, rank: {2:.3}".format(phrase_id, p.text, p.rank))
        print('--{}--'.format(p.text))
        
        unit_vector.append(p.rank)
        
        for chunk in p.chunks:
    #         print(" ", chunk.start, chunk.end)
            
            for sent_start, sent_end, sent_vector in sent_bounds:
                if chunk.start >= sent_start and chunk.start <= sent_end:
    #                 print(" ", sent_start, chunk.start, chunk.end, sent_end)
                    sent_vector.add(phrase_id)
                    break
    
        phrase_id += 1
    
        if phrase_id == LIMIT_PHRASES:  # берем N самых важных фраз
            break
    
    sum_ranks = sum(unit_vector)
    unit_vector = [rank/sum_ranks for rank in unit_vector]
            
    return unit_vector, sent_bounds


# In[ ]:


def print_summary(doc, unit_vector, sent_bounds, SUMMARY_SENT_NUMBER):
    '''
    Находится Евклидово расстояние от каждого предложения до "важного" вектора
    '''
    sent_rank = {}
    sent_id = 0
    
    for sent_start, sent_end, sent_vector in sent_bounds:
    #     print(sent_vector)
        sum_sq = 0.0
        
        for phrase_id in range(len(unit_vector)):
    #         print(phrase_id, unit_vector[phrase_id])
            
            if phrase_id not in sent_vector:
                sum_sq += unit_vector[phrase_id]**2.0
    
        sent_rank[sent_id] = sqrt(sum_sq)
        sent_id += 1
 
    #print(sorted(sent_rank.items(), key=itemgetter(1))[:15])
    
    sent_text = {}
    sent_id = 0
    
    for sent in doc.sents:
        sent_text[sent_id] = sent.text
        sent_id += 1
    
    num_sent = 0
    print('\nКлючевые предложения:')    
    for sent_id, rank in sorted(sent_rank.items(), key=itemgetter(1)):
        
        complete_sent = ''
    #     if len(sent_text[sent_id]) > 60:
        sent_tokens = nltk.sent_tokenize(sent_text[sent_id])
    #     print(sent_tokens)
        for s in sent_tokens:
            if (len(s) > 60) & ('http' not in s) & ('www.' not in s):
                complete_sent += s
        if complete_sent != '':
            print(sent_id, ': ',  complete_sent)
            num_sent += 1
        
        if num_sent == SUMMARY_SENT_NUMBER:
            break
            
    return sent_text


# In[ ]:


def show_summary(text):
    
    LIMIT_PHRASES, SUMMARY_SENT_NUMBER = limits(len(text))
    doc = prepare_doc(text)
    unit_vector, sent_bounds = top_phrases(doc, LIMIT_PHRASES)
    sent_text = print_summary(doc, unit_vector, sent_bounds, SUMMARY_SENT_NUMBER)
    
    return sent_text


# In[ ]:


def show_context(sent_id, sent_text):
    
    if sent_id not in ('', ' '):
        start_idx = int(sent_id)-2
        end_idx = int(sent_id)+3
        if start_idx < 0:
            start_idx = 0
        
        print('\nКонтекст: \n')
        art_text = []
        for key,value in sent_text.items():
            art_text.append(value)
        print(" ".join(art_text[start_idx:end_idx]))


# In[ ]:


def show_key_ph(text):
    
    LIMIT_PHRASES, SUMMARY_SENT_NUMBER = limits(len(text))
    doc = prepare_doc(text)
    unit_vector, sent_bounds = top_phrases(doc, LIMIT_PHRASES)

