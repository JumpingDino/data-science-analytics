import pandas as pd
import numpy as np

from sklearn.metrics import pairwise_distances


def apply_nomalization(x):
    x = np.array(x)
    
    vec_val = x**2
    mod = np.sum(x**2)
    
    return vec_val/mod


def similarity_topics(lda,
                      normalize = True,
                      used_metric = 'cosine'):
    
    ''' 
    Input : gensim Lda object posteriormente treinado com id2word tenha sido dado na instanciacao do objeto
    Output: pd.DataFrame Matriz de similaridade
    
    Calcula a matriz de similaridade dentre todos os tópicos'''
    
    topics_coordinates = lda.get_topics()
    word_names = lda.id2word.values()
    
    
    num_topics = lda.num_topics
    topic_names = ['topico_' + str(i) for i in range(0,num_topics)]

    print(f'Temos {num_topics} topicos formados')
    print(f'Temos {len(word_names)} palavras no nosso dicionario')
    
    topics_coordinates = pd.DataFrame(topics_coordinates,
                                      index = topic_names,
                                      columns = word_names)

    if(normalize == True):
        topics_coordinates = topics_coordinates.apply(apply_nomalization)

    return pd.DataFrame(pairwise_distances(topics_coordinates,metric = used_metric),
                 columns = topic_names,index = topic_names)