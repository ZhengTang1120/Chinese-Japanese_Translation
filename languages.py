import jieba
import MeCab
mecab = MeCab.Tagger ("-Owakati")
import json
from collections import defaultdict

k2c = json.load(open("k2c.json", encoding='utf-8'))
c2k = json.load(open("c2k.json", encoding='utf-8'))

class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS", 2: "UNK"}
        self.n_words = 3  # Count SOS, EOS and UNK

    def addSentence(self, sentence):
        if self.name == "japanese":
            for word in mecab.parse(sentence).split():
                if word not in k2c:
                    self.addWord(word)
            return mecab.parse(sentence).split()
        elif self.name == "chinese":
            for word in jieba.cut(sentence):
                self.addWord(word)
            return list(jieba.cut(sentence))
        else:
            for word in sentence.split():
                self.addWord(word)
            return sentence.split()

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1
