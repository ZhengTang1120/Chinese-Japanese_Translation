from model import *
from languages import *
import os
import time
import math
import argparse
from torch.nn.utils.rnn import pad_sequence

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def timeSince(since):
    now = time.time()
    s = now - since
    return '%s' % (asMinutes(s))

def split_sentence(sentence, name):
    if name == "japanese":
        return mecab.parse(sentence).split()
    elif name == "chinese":
        return seg.cut(sentence)
    else:
        return sentence.split()

def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] if word in lang.word2index else 2 for word in sentence]

def tensorFromIndexes(indexes):
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)

def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(1)
    return tensorFromIndexes(indexes)

def makeOutputIndexes(lang, output, input, pad_length):
    sourceset = {}
    id2source = {}
    pg_mat = np.ones((pad_length + 1, pad_length + 1)) * 1e-10
    for i, word in enumerate(input):
        if word not in sourceset:
            sourceset[word] = lang.n_words + len(sourceset)
            id2source[sourceset[word]] = word
        pg_mat[sourceset[word]-lang.n_words][i] = 1
    indexes = [0]

    for word in output:
        if word in sourceset:
            indexes.append(sourceset[word])
        elif word in lang.word2index:
            indexes.append(lang.word2index[word])
        elif word in k2c and k2c[word] in sourceset:
            indexes.append(sourceset[k2c[word]])
        else:
            indexes.append(2)

    indexes.append(1)
    return indexes, pg_mat, id2source

def get_pgmat(lang, input):
    sourceset = {}
    id2source = {}
    pg_mat = np.ones((len(input) + 1, len(input) + 1)) * 1e-10
    for i, word in enumerate(input):
        if word not in sourceset:
            sourceset[word] = lang.n_words + len(sourceset)
            id2source[sourceset[word]] = word
        pg_mat[sourceset[word]-lang.n_words][i] = 1
    return pg_mat, id2source

def predict(translator, sentences, input_lang, output_lang, max_length=MAX_LENGTH):
    translator.eval()
    for sentence in sentences:
        sentence = sentence[0]
        with torch.no_grad():
            input_tensor = tensorFromSentence(input_lang, sentence).view(-1, 1, 1)
            input_length = input_tensor.size()[0]
            pg_mat, id2source = get_pgmat(output_lang, sentence)
            pg_mat = torch.tensor(pg_mat, dtype=torch.float, device=device).unsqueeze(0)

            encoder_outputs, hidden = translator.encoder(input_tensor, 1)

            decoder_input = torch.tensor([[0]], device=device)  # SOS

            decoded_words = []

            for di in range(max_length):
                decoder_output, hidden, decoder_attention = translator.decoder(
                    decoder_input, hidden, encoder_outputs, pg_mat, 1)
                topv, topi = decoder_output.data.topk(1)
                if topi.item() == 1:
                    # decoded_words.append('<EOS>')
                    break
                else:
                    if topi.item() in output_lang.index2word: 
                        if topi.item() > 3:
                            decoded_words.append(output_lang.index2word[topi.item()])
                        else:
                            topv, topi = decoder_output.data[:,4:output_lang.n_words].topk(1)
                            decoded_words.append(output_lang.index2word[topi.item()+3])
                    elif topi.item() in id2source:
                        sourceword = id2source[topi.item()]
                        if sourceword in c2k:
                            decoded_words.append(c2k[sourceword])
                        else:
                            decoded_words.append(sourceword)
                    else:
                        print ("Error")


                decoder_input = topi.squeeze().detach()

            yield (''.join(decoded_words))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('corpra_dir')
    args = parser.parse_args()



    chi_lang = Lang("chinese")
    jap_lang = Lang("japanese")

    chi_lang_test = Lang("chinese")
    jap_lang_test = Lang("japanese")

    pairs = list()
    with open(args.corpra_dir+"/existing_parallel/segments.zh", encoding='utf-8') as fc, open(args.corpra_dir+"/existing_parallel/segments.ja", encoding='utf-8') as fj:
        c = fc.readlines()
        j = fj.readlines()
        for i in range(len(c)):
            pairs.append((chi_lang.addSentence(c[i]), jap_lang.addSentence(j[i])))
    
    test_sents = list()
    with open(args.corpra_dir+"/dev_dataset/segments.zh", encoding='utf-8') as fc, open(args.corpra_dir+"/dev_dataset/segments.ja", encoding='utf-8') as fj:
        c = fc.readlines()
        j = fj.readlines()
        for i in range(len(c)):
            test_sents.append((chi_lang_test.addSentence(c[i]), jap_lang_test.addSentence(j[i])))

    # batches = list(sort_and_batch(pairs, BATCH_SIZE))

    groups = group_via_length(pairs)

    batches = list()
    for l1 in groups:
        for l2 in groups[l1]:
            batch = list()
            for pair in groups[l1][l2]:
                if len(batch) < BATCH_SIZE:
                    batch.append(pair)
                else:
                    batches.append(batch)
                    batch = list()
            if len(batch)!=0:
                batches.append(batch)
    training_set = list()                 
    for batch in batches:
        chi_tensors = list()
        jap_tensors = list()
        pad_i_len = len(batch[0][0])
        pad_o_len = len(batch[0][1])
        pg_mats   = np.ones((len(batch), pad_i_len + 1, pad_i_len + 1)) * 1e-10
        for i, pair in enumerate(batch):
            chi_sent = pair[0]
            jap_sent = pair[1]
            chi_tensor = tensorFromSentence(chi_lang, chi_sent)
            jids, pg_mat, id2source = makeOutputIndexes(jap_lang, jap_sent, chi_sent, pad_i_len)
            jap_tensor              = tensorFromIndexes(jids)
            chi_tensors.append(chi_tensor)
            jap_tensors.append(jap_tensor)
            pg_mats[i] = pg_mat
        chi_tensors = torch.cat(chi_tensors,1).view(-1, len(batch), 1)
        jap_tensors = torch.cat(jap_tensors,1).view(-1, len(batch), 1)
        # chi_tensors = pad_sequence(chi_tensors, padding_value = 3)
        # jap_tensors = pad_sequence(jap_tensors, padding_value = 3)
        training_set.append((chi_tensors, jap_tensors, torch.tensor(pg_mats, dtype=torch.float, device=device)))
    print (len(training_set))
    learning_rate = 0.001
    hidden_size = 256

    encoder    = EncoderRNN(chi_lang.n_words, hidden_size).to(device)
    decoder    = AttnDecoderRNN(hidden_size, jap_lang.n_words, MAX_LENGTH, dropout_p=0.1).to(device)
    translator = Translator(encoder, decoder)

    optimizer = optim.Adam(translator.parameters())
    criterion = nn.NLLLoss()
    for epoch in range(20):
        translator.train()
        random.shuffle(training_set)
        
        total_loss = 0
        start = time.time()
        i = 0
        for input_tensor, target_tensor, pg_mat in training_set:
            i += 1

            print (i, input_tensor.size(1), input_tensor.size(0), target_tensor.size(0))
            
            output = translator(input_tensor, target_tensor, pg_mat)
            output_dim = output.shape[-1]
        
            output = output[1:].view(-1, output_dim)
            target_tensor = target_tensor[1:].view(-1)
            loss = criterion(output, target_tensor)
            loss.backward()

            clipping_value = 1#arbitrary number of your choosing
            torch.nn.utils.clip_grad_norm_(translator.parameters(), clipping_value)

            optimizer.step()
            # print(input_length, loss.item())
            total_loss += loss.detach().cpu().numpy()
            print (torch.cuda.memory_summary())
            torch.cuda.empty_cache()

        print (timeSince(start))
        print (total_loss)
        preds = predict(translator, test_sents[:10], chi_lang, jap_lang, max_length=100)
        os.mkdir("model/%d"%epoch)
        PATH = "model/%d"%epoch
        torch.save(encoder, PATH+"/encoder")
        torch.save(decoder, PATH+"/decoder")
        with open("model/%d"%epoch+"/preds.txt", "w") as f:
            for pred in preds:
                f.write(pred+'\n')