import jieba
import MeCab
mecab = MeCab.Tagger ("-Owakati")
import json
from collections import defaultdict
import pickle

k2c = json.load(open("k2c.json", encoding='utf-8'))
c2k = json.load(open("c2k.json", encoding='utf-8'))
MAX_LENGTH = 200
BATCH_SIZE = 32

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


def sort_and_batch(pairs, batch_size):
    pairs = sorted(pairs, key = lambda x: (len(x[0]), len(x[1])), reverse=True)
    batch = list()
    for pair in pairs:
        # if len(batch) == 0:
        #     batch.append([len(pair[0]), len(pair[1])])
        if len(batch) >= batch_size:#+1:
            yield batch
            batch = list()
        else:
            # batch[0][1] = max(len(pair[1]), batch[0][1])
            batch.append(pair)


def group_via_length(pairs):
    groups = defaultdict(dict)
    for pair in pairs:
        if len(pair[0])<=MAX_LENGTH and len(pair[1])<=MAX_LENGTH:
            if len(pair[1]) in groups[len(pair[0])]:
                groups[len(pair[0])][len(pair[1])].append(pair)
            else:
                groups[len(pair[0])][len(pair[1])] = [pair]
    return groups