from languages import *
import argparse

from nltk.translate.bleu_score import corpus_bleu

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('refs')
    parser.add_argument('preds')
    args = parser.parse_args()

    references = list()
    candidates = list()

    with open(args.preds) as f:
        for sentence in f:
            candidates.append(mecab.parse(sentence).split())

    with open(args.refs) as f:
        for sentence in f:
            references.append([mecab.parse(sentence).split()])
    print (len(references))
    print (len(candidates))
    print('Individual 1-gram: %f' % corpus_bleu(references, candidates, weights=(1, 0, 0, 0)))
    print('Individual 2-gram: %f' % corpus_bleu(references, candidates, weights=(0.5, 0.5, 0, 0)))
    print('Individual 3-gram: %f' % corpus_bleu(references, candidates, weights=(0.33, 0.33, 0.33, 0)))
    print('Individual 4-gram: %f' % corpus_bleu(references, candidates, weights=(0.25, 0.25, 0.25, 0.25)))