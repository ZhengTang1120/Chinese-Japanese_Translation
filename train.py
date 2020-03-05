from model import *
from languages import *
import random
import numpy as np
import os
import time
import math
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
    indexes = list()

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

def predict(encoder, decoder, sentences, input_lang, output_lang, max_length=MAX_LENGTH):
    for sentence in sentences:
        print (sentence[1])
        sentence = sentence[0]
        with torch.no_grad():
            input_tensor = tensorFromSentence(input_lang, sentence).view(-1, 1, 1)
            input_length = input_tensor.size()[0]
            pg_mat, id2source = get_pgmat(output_lang, sentence)
            pg_mat = torch.tensor(pg_mat, dtype=torch.float, device=device).unsqueeze(0)

            encoder_outputs, encoder_hidden = encoder(input_tensor, 1)

            decoder_input = torch.tensor([[0]], device=device)  # SOS

            decoder_hidden = (encoder_hidden[0].view(1, 1,-1), encoder_hidden[1].view(1, 1,-1))

            decoded_words = []

            for di in range(max_length):
                decoder_output, decoder_hidden, decoder_attention = decoder(
                    decoder_input, decoder_hidden, encoder_outputs, pg_mat, 1)
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

            yield (decoded_words)


if __name__ == '__main__':

    chi_lang = Lang("chinese")
    jap_lang = Lang("japanese")

    chi_lang_test = Lang("chinese")
    jap_lang_test = Lang("japanese")

    pairs = list()
    with open("existing_parallel/segments.zh") as fc, open("existing_parallel/segments.ja") as fj:
        c = fc.readlines()
        j = fj.readlines()
        for i in range(len(c)):
            if i >= 1000:
                break
            pairs.append((chi_lang.addSentence(c[i]), jap_lang.addSentence(j[i])))
    
    # test_sents = list()
    # with open("dev_dataset/segments.zh") as fc, open("dev_dataset/segments.ja") as fj:
    #     c = fc.readlines()
    #     j = fj.readlines()
    #     for i in range(len(c)):
    #         test_sents.append((chi_lang_test.addSentence(c[i]), jap_lang_test.addSentence(j[i])))

    batches = list(sort_and_batch(pairs, BATCH_SIZE))
    # for l1 in batches:
    #     for l2 in batches[l1]:
    #         print (l1, l2, len(batches[l1][l2]))
    # exit()
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
        # chi_tensors = torch.cat(chi_tensors,1).view(-1, len(batch), 1)
        # jap_tensors = torch.cat(jap_tensors,1).view(-1, len(batch), 1)
        chi_tensors = pad_sequence(chi_tensors, padding_value = 3)
        jap_tensors = pad_sequence(jap_tensors, padding_value = 3)
        training_set.append((chi_tensors, jap_tensors, torch.tensor(pg_mats, dtype=torch.float, device=device)))
            
    
    learning_rate = 0.0001
    hidden_size = 256

    encoder    = EncoderRNN(chi_lang.n_words, hidden_size).to(device)
    decoder    = AttnDecoderRNN(hidden_size, jap_lang.n_words, MAX_LENGTH, dropout_p=0.1).to(device)

    encoder_optimizer    = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer    = optim.Adam(decoder.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss(ignore_index=3)

    teacher_forcing_ratio = 0.5

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    for epoch in range(20):

        random.shuffle(training_set)
        total_loss = 0
        start = time.time()
        for input_tensor, target_tensor, pg_mat in training_set:
            loss = 0

            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()

            input_length  = input_tensor.size(0)
            target_length = target_tensor.size(0)
            batch_size    = input_tensor.size(1)

            encoder_outputs, encoder_hidden = encoder(input_tensor, batch_size)

            tensor = torch.tensor((), dtype=torch.int64, device=device)
            decoder_input = tensor.new_zeros((1, batch_size))
            decoder_hidden = (encoder_hidden[0].view(1, batch_size,-1), encoder_hidden[1].view(1, batch_size,-1))

            use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

            if use_teacher_forcing:
                # Teacher forcing: Feed the target as the next input
                for di in range(target_length):
                    decoder_output, decoder_hidden, decoder_attention = decoder(
                        decoder_input, decoder_hidden, encoder_outputs, pg_mat, batch_size)
                    # print (decoder_output.size())
                    # print (target_tensor[di].view(-1).size())
                    loss += criterion(decoder_output, target_tensor[di].view(-1))
                    decoder_input = target_tensor[di]  # Teacher forcing

            else:
                # Without teacher forcing: use its own predictions as the next input
                for di in range(target_length):
                    decoder_output, decoder_hidden, decoder_attention = decoder(
                        decoder_input, decoder_hidden, encoder_outputs, pg_mat, batch_size)

                    topv, topi = decoder_output.topk(1)
                    decoder_input = topi.squeeze().detach()  # detach from history as input
                    loss += criterion(decoder_output, target_tensor[di].view(-1))
                    # if decoder_input.item() == 1:
                    #     break


            loss.backward()

            # clipping_value = 1#arbitrary number of your choosing
            # torch.nn.utils.clip_grad_norm_(encoder.parameters(), clipping_value)
            # torch.nn.utils.clip_grad_norm_(decoder.parameters(), clipping_value)

            encoder_optimizer.step()
            decoder_optimizer.step()
            # print(input_length, loss.item())
            total_loss += loss.item() / target_length
        print (timeSince(start))
        print (total_loss)
        preds = predict(encoder, decoder, [pairs[71]], chi_lang, jap_lang, max_length=100)
        print (list(preds))
        # os.mkdir("model/%d"%epoch)
        # PATH = "model/%d"%epoch
        # torch.save(encoder, PATH+"/encoder")
        # torch.save(decoder, PATH+"/decoder")
        # with open("model/%d"%epoch+"/preds.txt", "w") as f:
        #     for pred in preds:
        #         f.write(pred+'\n')





