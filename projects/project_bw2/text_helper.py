import nltk
from nltk.tokenize import SpaceTokenizer
from nltk.stem import RSLPStemmer

import numpy as np

import pandas as pd

import re

from sklearn.feature_extraction.text import strip_accents_unicode
from string import punctuation

def tokenization(corpus ,stop_words = nltk.corpus.stopwords.words('portuguese')):
    
    '''Input : corpus é uma Serie de corpusumentos(frases)
       Output : Uma lista de listas com palavras 
    
    stop_words : lista de palavras que devem ser removidas
    '''
    
    #Tokenizacao
    spacetok = SpaceTokenizer()
    corpus = [spacetok.tokenize(phrases) for phrases in corpus]
    
    
    #stopwords
    if(stop_words != None):
        tmp_corpus = list()
        tmp_words = list()

        for phrases in corpus:
            for word in phrases:
                if(word not in stop_words):
                    tmp_words.append(word)
                else:
                    pass
            tmp_corpus.append(tmp_words)
            tmp_words = list()

        corpus = tmp_corpus
    else:
        pass
    
    return corpus

def entities_subs(corpus):
    '''Input : corpus é uma Serie de corpusumentos(frases)
       Output : Uma lista de listas com palavras 
       
       Substituicao de entidades reconhecidas por uma tag, desta forma conseguiremos diminuir o tamanho do nosso
       vocabilario
       
       #TODO: Além de substituir por tags, poderia ser interessante extrairmos a informação'''
    
    #regex_numbers 
    corpus = [re.sub(r'\d+','<NUM>', phrases) for phrases in corpus]
    
    #regex_links
    corpus = [re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
                        '<LINK>', phrases) for phrases in corpus]
    corpus = [re.sub('www.(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
                        '<LINK>', phrases) for phrases in corpus]

    #Reconhecimento de Entidades
    #TODO : NERs para extração de endereço, nomes, marcacoes, #s 
    
    return corpus

def text_correction(corpus):
    pass
    print('Correcao de texto nao realizada')
    return corpus

def collocations(corpus):
    pass
    print('Colocations nao realizada')
    return corpus

def stemming(corpus):
    '''Realiza Stemming com o RSLPStemmer'''
    rslps = RSLPStemmer()
    corpus = [rslps.stem(phrases) for phrases in corpus]
    return corpus

def lemmatization(corpus):
    pass
    print('Lemmatization nao realizada')
    return corpus


def rule_preprocessing(corpus,
                       stop_words = nltk.corpus.stopwords.words('portuguese'),
                      join_tokens = False,reduce_inflection = 'stemming'):
    
    '''Pre processamento atraves de regras, basta inserirmos uma pd.Series na imput e as regras pre-estabelecidas
    melhorarao o texto'''
    
    #Removendo os espacos antes e depois da frase e substituicao de multiplos espacos
    corpus = [phrases.strip() for phrases in corpus]
    corpus = [re.sub(' +',' ', phrases) for phrases in corpus]
    
    #coloca todas as palavras em minúsculo
    corpus = [phrases.lower() for phrases in corpus]
    
    #extrai pontuacoes    
    corpus = [phrases.translate(str.maketrans('', '', punctuation)) for phrases in corpus]    
    
    #Substituicao de entidades
    corpus = entities_subs(corpus)
    
    #Correcao palavras
    corpus = text_correction(corpus)
    
    #retira acentos
    corpus = [strip_accents_unicode(phrases) for phrases in corpus]
    
    #tokenizacao e remocao de stop_words
    corpus = tokenization(corpus ,stop_words = stop_words)
    
    ###################### ESTA PARTE PRECISA MELHORAR #########################
    
#     #Reduzir inflexoes das palavras
#     if(reduce_inflection == 'stemming'):
#         corpus =  [stemming(phrases) for phrases in corpus]
        
#     elif(reduce_inflection == 'lemmatization'):
#         corpus = lemmatization(corpus)
        
#     else:
#         pass   
    
    ###################### ESTA PARTE PRECISA MELHORAR #########################
    
    #junta as palavras tokenizadas
    if(join_tokens == True):
        corpus = [' '.join(phrases) for phrases in corpus]
        
    return corpus