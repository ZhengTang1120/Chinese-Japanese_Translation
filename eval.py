from model import *
from languages import *
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('epoch')
    args = parser.parse_args()

    chi_lang = Lang("chinese")
    jap_lang = Lang("japanese")

    chi_lang_test = Lang("chinese")
    jap_lang_test = Lang("japanese")

    pairs = list()
    with open("existing_parallel/segments.zh", encoding='utf-8') as fc, open("existing_parallel/segments.ja", encoding='utf-8') as fj:
        c = fc.readlines()
        j = fj.readlines()
        for i in range(len(c)):
            pairs.append((chi_lang.addSentence(c[i]), jap_lang.addSentence(j[i])))

    test_sents = list()
    with open("dev_dataset/segments.zh", encoding='utf-8') as fc, open("dev_dataset/segments.ja", encoding='utf-8') as fj:
        c = fc.readlines()
        j = fj.readlines()
        for i in range(len(c)):
            test_sents.append((chi_lang_test.addSentence(c[i]), jap_lang_test.addSentence(j[i])))

    PATH = "model/%d"%args.epoch
    encoder = torch.load(PATH+"/encoder")
    decoder = torch.load(PATH+"/decoder")
    translator = Translator(encoder, decoder)

    preds = predict(translator, test_sents, chi_lang, jap_lang, max_length=100)

    with open("preds.txt", "w") as f:
            for pred in preds:
                f.write(pred+'\n')