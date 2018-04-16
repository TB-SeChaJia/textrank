#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import, unicode_literals
import sys,re
import pandas as pd
from tqdm import tqdm
from operator import itemgetter
from collections import defaultdict

def readlines(path):

    with open(path) as rf:
        lines = rf.readlines()
        lines = [line.strip() for line in lines if line.strip()]
        return lines


itervalues = lambda d: iter(d.values())
tokenizer = lambda x:re.split("\s+",x)

class KeywordExtractor(object):
        
    STOP_WORDS = set(("དང","ཀྱི","གི","གྱི","ཡི","འི","ཀྱིས","གིས","གྱིས","ཡིས","འིས","སུ","རུ",
                      "ཏུ","དུ","ན","ལ","ར","རའ","ཏེ","དེ","སྟེ","ནི","ཀྱང","འང","ཡང","ཅིང",
                      "ཞིང","ཤིང","ཅེ་ན","ཞེ་ན","ཤེ་ན","ཅེས་པ","ཞེས་པ","ཅེའོ","ཞེའོ","ཤེའོ","ཅིག","ཤིག",
                      "ཞིག","ཡོད","འདུག","བདའ","ལྡན","གནས","ཡིན","མིན","རེད","མ","མི","མེད",
                      "མིན","རྒྱུ","བཞིན","གང","ཅི","ཇི","ཟེར","ཞུ","ཞུས","མཛོད","རབ་ཏུ","ཀུན་ནས",
                      "ཚུལ","གཅིག","གཉིས","གསུམ","བཞི","ལྔ","དྲུག","བདུན","བརྒྱད","དགུ","བཅུ",
                      "བཅུ་གཅིག","བཅུ་གཉིས","བཅུ་གསུམ","བཅུ་བཞི","བཅོ་ལྔ","བཅུ་དྲུག","བཅུ་བདུན","བཅོ་བརྒྱད",
                      "བཅུ་དགུ","ཉི་ཤུ","སུམ་ཅུ","བཞི་བཅུ","ལྔ་བཅུ","དྲུག་ཅུ","བདུན་ཅུ","བརྒྱད་ཅུ","དགུ་བཅུ","བརྒྱ",
                      "སྟོང","ཁྲི","འབུམ","ས་ཡ","ལས","ནས","སོགས","བཅས","རྣམས","ཕྱིར","ཁོ་ན",
                      "འདི","དེ","ཡེ་གནས","ཀུན","སྣང","ངང","ཡོངས","ནང","ཕྱི","སྟེང","འོག","སླད",
                      "ཆེད","ཆེད་དུ","སླད་དུ","ལ་སོགས","བྱེད་པ","གང་ཡིན","གོ","ངོ","དོ","ནོ","བོ","མོ",
                      "འོ","རོ","ལོ","སོ","ཏོ","མེད་པ","དག","འདྲ","རིམ་བཞིན","མོད","དེ་ལྟ","ཇི་ལྟར",
                      "དེ་ནས","མཚུངས","ཞེས","ཅེས","མང","ཉུང","ལྟར","།","།།"))

    def set_stop_words(self, stop_words_path):
        if not os.path.isfile(stop_words_path):
            raise Exception("file does not exist: " + stop_words_path)

        lines = readlines(stop_words_path)
        for line in lines:
            self.STOP_WORDS.add(line)


class UndirectWeightedGraph:
    d = 0.85

    def __init__(self):
        self.graph = defaultdict(list)

    def addEdge(self, start, end, weight):
        # use a tuple (start, end, weight) instead of a Edge object
        self.graph[start].append((start, end, weight))
        self.graph[end].append((end, start, weight))

    def rank(self,steps=20):
        ws = defaultdict(float)
        outSum = defaultdict(float)

        wsdef = 1.0 / (len(self.graph) or 1.0)
        for n, out in self.graph.items():
            ws[n] = wsdef  #ཐོག་མར་མཐའ་རེ་རེར་གཅིག་གྱུར་གྱིས་སྲིད་ཕྱོད་སྟེར་བ།
            outSum[n] = sum((e[2] for e in out), 0.0)

        # 
        sorted_keys = sorted(self.graph.keys())
        for x in range(steps):  # 10 iters
            for key in sorted_keys:
                s = 0
                for e in self.graph[key]:
                    s += e[2] / outSum[e[1]] * ws[e[1]]
                ws[key] = (1 - self.d) + self.d * s

        (min_rank, max_rank) = (sys.float_info[0], sys.float_info[3])

        for w in itervalues(ws):
            if w < min_rank:
                min_rank = w
            if w > max_rank:
                max_rank = w

        for n, w in ws.items():
            # to unify the weights, don't *100.
            ws[n] = (w - min_rank / 10.0) / (max_rank - min_rank / 10.0)

        return ws


class TextRank(KeywordExtractor):

    def __init__(self):
        self.tokenizer = tokenizer # མིང་དུམ་གཅོད་བྱེད་པ།
        self.stop_words = self.STOP_WORDS.copy()  # མཁོ་མེད་མིང་གི་ཚོགས་པ།
        self.span = 5 # གོམ་ཐག

    def pairfilter(self, word):
        # ཚེག་ཁྱིམ་གཅིག་ཅན་དང་མཁོ་མེད་མིང་སོགས་བསུབ་པ།
        return (word not in self.stop_words) #word.count("་") >= 1 and 

    def textrank(self, sentence, topK=3, withWeight=False):
        """
        Extract keywords from sentence using TextRank algorithm.
        Parameter:
            - sentcence: ཐག་གཅོད་བྱའི་ཚིག
            - topK: མིང་ག་ཚོད་ཅིག་ཕྱིར་མངོན།
            - withWeight: མིང་དང་སྲིད་ཕྱོད་གཉི་ཀ་ཕྱིར་མངོན་ནམ།
        """

        g = UndirectWeightedGraph()
        cm = defaultdict(int)
        words = tuple(self.tokenizer(sentence))
        for i, word in enumerate(words):
            if self.pairfilter(word):
                for j in range(i + 1, i + self.span):
                    if j >= len(words):
                        break
                    if not self.pairfilter(words[j]):
                        continue
                    cm[(word, words[j])] += 1

        for terms, w in cm.items():
            g.addEdge(terms[0], terms[1], w)
        nodes_rank = g.rank()
        if withWeight:
            tags = sorted(nodes_rank.items(), key=itemgetter(1), reverse=True)
        else:
            tags = sorted(nodes_rank, key=nodes_rank.__getitem__, reverse=True)

        if topK:
            return tags[:topK]
        else:
            return tags

def get_keys(texts,topk=3):
    
    keyList = []
    for text in tqdm(texts):
        TR = TextRank()
        keys = TR.textrank(text)
        keyList.append(" ".join(keys))
        
    return keyList
    
    
if __name__ == "__main__":     
    
    """
    གལ་ཏེ་ཡིག་ཁུག csv ཅན་གྱི་གཞི་གྲངས་ཡོད་ན་འདིས་ཐག་གཅོད་བྱ་ཆོག་གོ
    dataFile = "data.csv"
    datas = pd.read_csv(dataFile)
    texts = datas["text"]
    keys  = get_keys(texts)
    
    ids = list(range(1,len(keys)+1))
    #print(len(idList),len(keyList),len(textList))
    result = pd.DataFrame({"id": ids, "key": keys, "text": texts},
                         columns=['id','key',"text"])
    result.to_csv("keys_data.csv",index=False)
    """
    TR = TextRank()
    keys = TR.textrank("ང ནི སློབ་མ ཞིག ཡིན ། ཁོ ནི ཡང སློབ་མ ཞིག ཡིན ། མོ ནི དགེ་རྒན ཞིག ཡིན")
    print(keys)
    
    
