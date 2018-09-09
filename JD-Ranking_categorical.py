# -*- coding: utf-8 -*-
"""
Spyder Editor

Este é um arquivo de script temporário.
"""

import pandas as pd
import numpy as np
import math

# =============================================================================
# #Ent : Calcula a entropia a partor de uma lista de probabilidades,quanto maior o 
# #valor , de ENT, mais "randômico" é o fenômeno
#     
# #left_prob_terms: Calcula os termos de probabilidade em E(S,T)
#     
# #right_ent_terms: Calcula os termos de entropia em E(S,T)
#  
# #final_product : Realiza o produto de left_* e right_* e retorna a soma dos elementos
#     
# #Init_Ent : Dado  y , calcula a entropia inicial do meu dataframe 
#     
# #RankMyCats : Dado x e y , calcula-se o ranking das minhas variaveis explicativas
# #categóricas,quanto menor o valor de EntropyBySplit mais interessante é a variável
# #pois máximiza o ganho de informação,lembre-se que Gain = Init_Ent - EntropyBySplit
# 
# =============================================================================

#data = pd.read_csv("Hunt.csv")

#y=data["Crédito"]
#X = data.drop(['Crédito'],axis=1)
#maxCount = max(data[y].value_counts())
#indexes = [i for i,x in enumerate(data[y].value_counts()) if x == maxCount][0]
#default = data[y].value_counts().index[indexes]

def Ent(list_pi):
    #list_pi = list(list_pi)
    #Gambiarra para lidar com pi =0 -> log 0 
    for i in range(len(list_pi)):
        if (list_pi[i] < 1e-12):
            list_pi[i]= 1.0
    EntValues = [-i*math.log2(i) for i in list_pi]
    finalVal = sum(EntValues)
    return finalVal

def left_prob_terms(ListOfFreqTab):
    left_prob_term_list = []
    for it in ListOfFreqTab:
        occur = it.sum(axis=1).sum()
        Table_to_pi_list= [(it.sum(axis=1).reset_index()[0][k]/occur) for k in range(it.shape[0])] 
        left_prob_term_list.append(Table_to_pi_list)
    return left_prob_term_list


def right_ent_terms(ListOfFreqTab):
    right_prob_term_list = []
    right_ent_terms = []
    for it in ListOfFreqTab:
        Table_to_pi_list= [(it.iloc[c,:]/sum(it.iloc[c,:])) for c in range(it.shape[0])] 
        right_prob_term_list.append(Table_to_pi_list)
        
    for i in range(len(right_prob_term_list)):
        inter_list = []
        for k in range(len(right_prob_term_list[i])):
            inter_list.append(Ent(right_prob_term_list[i][k]))
        right_ent_terms.append(inter_list)
    return right_ent_terms

def final_product(left,right):
    finalvec = list()
    if(len(left) != len(right)):
        print("O tamanho das LISTAS inseridas são diferentes")
    else:
        for i in range(len(left)):
            finalvec_value = np.array(left[i]) * np.array(right[i])
            final_value_term = sum(finalvec_value)
            finalvec.append(final_value_term)
    return finalvec

def Init_Ent(y):
    c = len(y.value_counts()) #vê o nº de classes distintas em y
    p_i = [ (y.value_counts()[i]/len(y)) for i in range(c)] #calcula a prob de cada classe
    
    Ent(p_i)# Calcula a entropia da label_y
    
    Initial_Entropy = Ent(p_i)
    print("A Entropia inicial é dada por ",Initial_Entropy)

def RankMyCats(X,y):
    FreqTab_list = [pd.crosstab(X[j],y) for j in X.columns]
    left = left_prob_terms(ListOfFreqTab)
    right = right_ent_terms(ListOfFreqTab)
    ent_by_label = final_product(left,right)
    myRankingDf = pd.DataFrame(list(zip(X.columns,finalmente)),columns = ['Label','EntropyBySplit'])
    return myRankingDf
