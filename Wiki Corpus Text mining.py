#!/usr/bin/env python
# coding: utf-8

# Text mining on Wiki Corpus to extract Named Entities

# Importing NLTK Natural Library Toolkit

import nltk


#Opening Wiki Corpus text file in read mode and removing new line

with open('wiki_corpus.txt','r') as myfile:
    data = myfile.read().replace('\n','')


# Removing slash

data2=data.replace("/","")

#  Checking the data if the newline is removed or not

 for i, line in enumerate(data2.split("\n")):
     if i>10:
         break
     print(str(i) +':\t' + line)

# Importing both the tokenizers from nltk library 

from nltk import sent_tokenize, word_tokenize


# Applying word tokenizer on data

word_tokenize(data2)


#  Applying Sentence tokenizer on Data

sent_tokenize(data2)


# Importing stopwords and storing the english stopwords in a variable 

from nltk.corpus import stopwords
stopwords_en = stopwords.words("english")


# Applying lower function after word tokenization on data

single_tokenized_lowered = list(map(str.lower, word_tokenize(data2)))
print(single_tokenized_lowered)


# Priniting words which are not in stopwords

stopwords_en = set(stopwords.words("english"))
print([word for word in single_tokenized_lowered if word not in stopwords_en])


# Importing & printing punctuation 

from string import punctuation 
print(type(punctuation), punctuation )


#Union of stopwords along with punctuation

stopwords_en_withpunct = stopwords_en.union(set(punctuation))
print(stopwords_en_withpunct)


#printing words which are not in punctuation 

print([word for word in single_tokenized_lowered if word not in punctuation])


#Applying stemming using Porter Stemmer

from nltk.stem import PorterStemmer
porter = PorterStemmer()

for word in single_tokenized_lowered:
    print(porter.stem(word))


#Applying lemmatization using Word Net Lemmatizer

from nltk.stem import WordNetLemmatizer
wnl=WordNetLemmatizer()


for word in single_tokenized_lowered:
    print(wnl.lemmatize(word))


# Tagging Part of speech (POS) in words

stop_words = set(stopwords.words('english'))
tokenized = sent_tokenize(data2)
for i in tokenized:
    wordslist = nltk.word_tokenize(i)
    word = [w for w in wordslist if  not w  in stop_words ]
    tagged = nltk.pos_tag(wordslist)
    print(tagged)


# Named Entity Recognition and Extraction

sentences = nltk.sent_tokenize(data2)
tokenized_sentences = [nltk.word_tokenize(sentence) for sentence in sentences]
tagged_sentences = [nltk.pos_tag(sentence) for sentence in tokenized_sentences]
chunked_sentences = nltk.ne_chunk_sents(tagged_sentences, binary = True)

def extract_entity_names(t):
    entity_names = []
    
    if hasattr(t, 'label') and t.label:
        if t.label() == 'NE':
            entity_names.append(' '.join([child[0] for child in t]))
        else:
            for child in t:
                entity_names.extend(extract_entity_names(child))
    return entity_names
entity_names = []
for tree in chunked_sentences:
    entity_names.extend(extract_entity_names(tree))
print(set(entity_names))



