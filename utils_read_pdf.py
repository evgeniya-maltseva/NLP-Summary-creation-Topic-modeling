#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import textract
import nltk, re
import string
import pandas as pd
from tqdm import tqdm
import utils
from difflib import SequenceMatcher


# In[ ]:


SOURCE_FOLDER = 'Source_data/'
extra = ['References', 'Further reading', 'Further information', 'Figure notes', 
         'Table of Contents', 'Contents', 'CONTENTS', 'Contact information']


# In[ ]:


# Только буквы
def only_word_tokenize(text):

#     lemmatizer = WordNetLemmatizer()
#     text = clean(text)
    tokens = nltk.word_tokenize(text)
    tokens = [token for token in tokens if re.match(r"[a-z]|[A-Z]", token)] #^\d*[a-z]
#     tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    return tokens

# Большие буквы
def tokenize_caps(text): 

#     lemmatizer = WordNetLemmatizer()
#     text = clean(text)
    tokens = nltk.word_tokenize(text)
    tokens = [token for token in tokens if re.match(r"[A-Z]", token)] #^\d*[a-z]
#     tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    return tokens

# Только цифры
def tokenize_numbers(text):

#     lemmatizer = WordNetLemmatizer()
#     text = clean(text)
    tokens = nltk.word_tokenize(text)
    tokens = [token for token in tokens if re.match(r"\d", token)]
#     tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    return tokens


# In[ ]:


def read_pdf(source_folder, filenames):
    articles = []

    for f in tqdm(filenames):
        text = textract.process(source_folder + f + '.pdf', method='pdfminer', encoding = 'ISO-8859-1')
        text = str(text)
        text = text.replace('\\n', ' ')
        text = re.sub(r'\\xab|\\t|\\xa0|\\xb3', '', text) #|\\xa9
        articles.append(text)
        
    return articles


# In[ ]:


#TODO sent_tokenize 2 times
class Running_title:
    
    def __init__(self, text):
        
        self.text = text
    
    def find_running_titles(self, kind = 'up'):
        
        pages = [utils.sent_tokenize(page) for page in self.text]
        del pages[-1]
        
        
        if kind == 'up':
            sent_idx = 0
        else:
            sent_idx = -1
        sent_list = []
        
        for i in range(len(pages)):
            try:
                sent_list.append(pages[i][sent_idx])
            except IndexError:
                pass
        
        run_title = []
        for i in range(len(sent_list)):
            
            string1 = sent_list[i]
            try:
                string2 = sent_list[i+1]
                string3 = sent_list[i+2]
            except IndexError:
                string2 = sent_list[0]
                string3 = sent_list[0]
                
            match = SequenceMatcher(None, string1, string2).find_longest_match(0, len(string1), 0, len(string2))
            match2 = SequenceMatcher(None, string1, string3).find_longest_match(0, len(string1), 0, len(string3))
    #         print(match)  
            run_title.append(string1[match.a: match.a + match.size])
            run_title.append(string1[match2.a: match2.a + match2.size]) 
#             print('  S1', string1)
#             print('  S2', string2)
#             print('\n', string1[match.a: match.a + match.size], '\n') 
#             print('\n', string1[match2.a: match2.a + match2.size], '\n')
        
        run_title = [rt for rt in set(run_title) if len(rt) > 25]
        run_title = [rt for rt in set(run_title) if rt[0].isupper()]
        run_title.sort(key=len, reverse = True)        
        
        return run_title, pages
    
    
    def delete_run_title(self, sentence, run_title):
        
        for rt in set(run_title):
            
            s = sentence.replace(rt, '')
            
            sentence = s
        
        return sentence
    
    def clean_doc(self):
        
        run_title_up, pages = self.find_running_titles(kind = 'up')
        run_title_down, _ = self.find_running_titles(kind = 'down')
        
        for p in pages:
            if p != []:
                p[0] = self.delete_run_title(p[0], run_title_up)
                p[-1] = self.delete_run_title(p[-1], run_title_down)
                
        clean_text = []
        for page in pages:
            whole_page = ' '.join(page)
            clean_text.append(whole_page)
                
        return clean_text


# In[ ]:


def clean_pdf(text):

    text = text.split('\\x0c')
    
    rt_instance = Running_title(text)
    text = rt_instance.clean_doc()
#     print(text)
    
    clean_text = ''
    pages = {}
    for i in range(len(text)):
    
        page = text[i]
        
        for term in extra:
            if term in page:
                
                page = page.split(term)[0]
                
        sent_len = []
        caps_len = []
        numb_len = []
        sentences = nltk.sent_tokenize(page)
#         print(sentences)
        for s in sentences:
            tokens = only_word_tokenize(s)
#             print('',tokens)
            sent_len.append(len(tokens))
            
            tokens_caps = tokenize_caps(s)
            caps_len.append(len(tokens_caps))
#             print('',tokens_caps)
            
            tokens_numb = tokenize_numbers(s)
            numb_len.append(len(tokens_numb))
#             print('',tokens_numb)
        
        try:
            max_words = max(sent_len) 
            max_caps = max(caps_len)
            max_numb = max(numb_len)
            mean_caps = round(sum(caps_len)/len(caps_len))
            mean_numb = round(sum(numb_len)/len(numb_len))
        except ValueError:
            max_words, max_caps, max_numb, mean_caps, mean_numb = 0, 0, 0, 0, 0
        
#         print('Mean words: ', round(sum(sent_len)/len(sent_len)))
#         print('Max words: ', max_words)
        
#         print('Mean capital letters: ', mean_caps)
#         print('Max capital letters: ', max_caps)
        
#         print('Mean numbers: ', mean_numb)
#         print('Max numbers: ', max_numb)
        
        if (max_words > 150) | ((max_caps > 20) & (mean_caps > 7)) | ((max_numb > 12) & (mean_numb > 2)):
            pass
        else:
            clean_text += page
#             print(i)
#             print(page)
        
#         pages[i] = [round(sum(sent_len)/len(sent_len)), max(sent_len), 
#                     round(sum(caps_len)/len(caps_len)), max(caps_len),
#                     round(sum(numb_len)/len(numb_len)), max(numb_len)]
        clean_text = re.sub(r'([A-Z]\s){2,}', '', clean_text)
        
    return clean_text

