#!/usr/bin/python
# -*- coding:utf-8 -*-

'''
Utilities for parsing textRank algorithm files. 
=======================
author: bsi chrb skybs    
date:2016-04-17

དྲྭ་རྐུན།  བསེ་ཁྲབ་སྐྱབས།
དུས་ཚོད། ༢༠༡༧ ༠༤ ༡༧
=======================
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import numpy as np
import math


def dtm(docs, min = 0):
    #མིང་ཚིག་རེ་རེའི་ཟློས་གྲངས་བརྩི་དགོས།
    wordList = sum(docs,[])
    # འདེིར་ཟློས་ཕྱོད་ཆེ་དྲག་པའམ་ཕྲད་ལ་སོགས་ཀྱི་དོན་སྟོན་མི་ནུས་བའི་མིང་ཚིག་དག་དོར་བ།
    counter_word = collections.Counter(wordList)
    #ཆེ་ནས་ཆུང་བའི་གོ་རིམ་ལྟར་བསྒྲིག་དགོས།
    count_pairs_word = sorted(counter_word.items(), key=lambda x: (-x[1], x[0]))
    count_pairs_word = [(w,i) for w,i in count_pairs_word if i > min]
    vocabList, _ = list(zip(*count_pairs_word))
    
    nDoc = len(docs)
    nVocab = len(vocabList)
    dtm = np.zeros((nDoc, nVocab))
    
    row_idx = 0
    for line in docs:
        for word in line:
            idx = vocabList.index(word)
            dtm[row_idx, idx] += 1.0
        row_idx +=1
    return dtm, vocabList

def tfIdf(dtm):
    nDoc = dtm.shape[0]
    nTerm = dtm.shape[1]
    # ཟློས་གྲངས་དེ་ཟློས་ཕྱོད་དུ་སྒྱུར་བ། དེའང་མིང་ཚིག་རེ་རེ་ཚིག་གཅིག་ཏུ་བྱུང་བའི་ཟློས་ཕྱོད་ཡིན།
    dtmNorm = dtm/dtm.sum(axis=1, keepdims=True)
    dtmNorm = np.nan_to_num(dtmNorm)
    tfIdfMat = np.zeros((nDoc,nTerm))
    
    for j in range(nTerm):
        tfVect = dtmNorm[:, j]
        # ཚིག་ཚང་མའི་ཁྲོད་ཀྱི་མིང་ཚིག་གི་སྡོམ་གྲངས་བརྩི་དགོས།
        nExist = np.sum(tfVect > 0.0) # if tfVect is 0.0, word is not in current doc
        idf = 0.0
        # int32
        if (nExist > 0):
            idf = np.log(nDoc/nExist)/np.log(2) # log2()
        else:
            idf = 0.0
        tfIdfMat[:,j] = tfVect * idf
  
    return tfIdfMat

def cosine_similarity(v1,v2):
    # ཕྱོགས་ཚད་གཉིས་ཀྱི་བཙུར་ཟུར་རྩི་བ།  ཉེ་མཚུངས་ཚད་རྩི་བ།
    # "compute cosine similarity of v1 to v2: (v1 dot v2)/{||v1||*||v2||)"
    sumxx, sumxy, sumyy = 0.0, 0.0, 0.0
    sumxx = np.dot(v1, v1.T)
    sumyy = np.dot(v2, v2.T)
    sumxy = np.dot(v1, v2.T)
    cosine = 0.0
    if (math.sqrt(sumxx*sumyy) == 0.0):
        cosine = 0.0
    else:
        cosine = sumxy/math.sqrt(sumxx*sumyy)
    
    return cosine


def calcAdj(tfIdfMat):
    nDoc = tfIdfMat.shape[0]
    simMat = np.zeros((nDoc, nDoc))
    adjMat = np.zeros((nDoc, nDoc))
    for i in range(nDoc):
        for j in range(nDoc):
            v1 = tfIdfMat[i,:]
            v2 = tfIdfMat[j,:]
            cosine_sim = cosine_similarity(v1, v2)
            simMat[i, j] = cosine_sim
    
    # ཆ་སྙོམས་ཀྱི་ཉེ་མཚུངས་ཚད། ཉེ་མཚུངས་ཚད་ཆུང་དྲག་ན་དོར་བ། ཚད་ལོངས་ན་གཅིག་གིས་མཚོན་པ།
    mean_sim = np.mean(simMat)
    for i in range(nDoc):
        for j in range(nDoc):
            if (simMat[i,j] > mean_sim):
                adjMat[i, j] = 1.0
    return adjMat

def pagerank(nDim, adjMat, alpha, itermax):
    '''
    Args:
    alpha: damping factor, 
    itermax: iteration Number
    '''
    P = np.ones((nDim, 1)) * (1/nDim)
    
    # normalize adjacency Matrix
    B = adjMat/adjMat.sum(axis=1, keepdims=True)
    B = np.nan_to_num(B)
    
    U = np.ones((nDim, nDim)) * (1/nDim)
    
    M = alpha * B + (1-alpha) * U
    
    for i in range(itermax):
        P = np.dot(M.T, P)
    score = P.tolist()
    return P

def rank(docs, percent=0.2, alpha=0.85,itermax=500,order_by = 'id'):
    '''
    docs: input list of list, separated words;
    percent: percent of sentences to keep
    order_id: sorted 
    alpha: 
    itermax: the number of iterm
    '''
    
    nDoc = len(docs)
    docTermMat, vocab = dtm(docs)
    tfIdfMat = tfIdf(docTermMat)
    adjMat = calcAdj(tfIdfMat)
    score = pagerank(nDoc, adjMat, alpha, itermax)
    
    docScore = []
    for i in range(nDoc):
        docScore.append((i, docs[i], score[i])) # return list of tuples [(index, doc, score)]
    
    # Ranking
    sortedList = sorted(docScore, key=lambda item : -item[2]) # sort to desc order of score
    nKeep = np.int(nDoc * percent)
    doc_sum = sortedList[0: nKeep] # doc_sum: nKeep number of document highest score
    
    if (order_by == 'score') :
        rank_idx = 2
    elif (order_by == 'id'):
        rank_idx = 0
    else :
        rank_idx = 0
    doc_sum_rerank = sorted(doc_sum, key=lambda item : item[rank_idx]) # nKeep number of document sort to increase order of index
    return doc_sum_rerank

if __name__ == '__main__':
  
    # document term Matrix
    docs = [['དེ་རིང','གནམ་གཤིས','ཧ་ཅང', 'བཟང'],['གནམ་གཤིས', 'ཡག'],['དེ་རིང'],['ཁྱོད', 'ནི', 'སུ', 'སུ', 'ཡིན']]
    percent = 0.5
    list = rank(docs, percent)
    print(list)

