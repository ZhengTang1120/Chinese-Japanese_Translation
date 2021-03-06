from model import *
from languages import *
import random
import os
import time
import math
import numpy as np

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
    return [lang.word2index[word] for word in sentence]

def tensorFromIndexes(indexes):
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)

def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(1)
    return tensorFromIndexes(indexes)

def makeOutputIndexes(lang, output, input):
    sourceset = {}
    id2source = {}
    pg_mat = np.ones((len(input) + 1, len(input) + 1)) * 1e-10
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

def evaluate(encoder, decoder, sentence, input_lang, output_lang, max_length=100):
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, sentence)
        input_length = input_tensor.size()[0]
        pg_mat, id2source = get_pgmat(output_lang, sentence)
        pg_mat = torch.tensor(pg_mat, dtype=torch.float, device=device)

        encoder_output, encoder_hidden = encoder(input_tensor)
        encoder_outputs  = encoder_output.view(input_length, -1)

        decoder_input = torch.tensor([[0]], device=device)  # SOS

        decoder_hidden = (encoder_hidden[0].view(1, 1,-1), encoder_hidden[1].view(1, 1,-1))

        decoded_words = []

        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs, pg_mat)
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == 1:
                # decoded_words.append('<EOS>')
                break
            else:
                if topi.item() in output_lang.index2word and topi.item() > 2:
                    decoded_words.append(output_lang.index2word[topi.item()])
                elif topi.item() in id2source:
                    sourceword = id2source[topi.item()]
                    if sourceword in c2k:
                        decoded_words.append(c2k[sourceword])
                    else:
                        decoded_words.append(sourceword)
                else:
                    topv, topi = decoder_output.data[3:output_lang.n_words].topk(1)
                    decoded_words.append(output_lang.index2word[topi.item()])


            decoder_input = topi.squeeze().detach()

        return decoded_words


if __name__ == '__main__':

    chi_lang = Lang("chinese")
    jap_lang = Lang("japanese")

    pairs = list()

    with open("existing_parallel/segments.zh") as fc, open("existing_parallel/segments.ja") as fj:
        c = fc.readlines()
        j = fj.readlines()
        n = 0
        for i in range(len(c)):
            if n<1000:
                n+=1
                pairs.append((chi_lang.addSentence(c[i]), jap_lang.addSentence(j[i])))

    print (jap_lang.n_words)
    print (chi_lang.n_words)

    training_set = list()

    for pair in pairs:
        chi_sent = pair[0]
        jap_sent = pair[1]
        chi_tensor = tensorFromSentence(chi_lang, chi_sent)
        jids, pg_mat, id2source = makeOutputIndexes(jap_lang, jap_sent, chi_sent)
        jap_tensor              = tensorFromIndexes(jids)
        training_set.append((chi_tensor, jap_tensor, torch.tensor(pg_mat, dtype=torch.float, device=device), id2source))
    learning_rate = 0.001
    hidden_size = 256

    encoder    = EncoderRNN(chi_lang.n_words, hidden_size).to(device)
    decoder    = AttnDecoderRNN(hidden_size, jap_lang.n_words, 100, dropout_p=0.1).to(device)

    encoder_optimizer    = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer    = optim.Adam(decoder.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()

    teacher_forcing_ratio = 0.5

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    test_sent = pairs[71]
    # print(test_sent[0])
    # print(test_sent[1])
    # print (makeOutputIndexes(jap_lang, test_sent[1], test_sent[0])[0])

    for epoch in range(20):

        random.shuffle(training_set)
        total_loss = 0
        start = time.time()
        for input_tensor, target_tensor, pg_mat, id2source in training_set:
            loss = 0

            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()

            input_length = input_tensor.size(0)
            target_length = target_tensor.size(0)

            encoder_output, encoder_hidden = encoder(input_tensor)
            encoder_outputs  = encoder_output.view(input_length, -1)

            decoder_input = torch.tensor([[0]], device=device)
            decoder_hidden = (encoder_hidden[0].view(1, 1,-1), encoder_hidden[1].view(1, 1,-1))

            use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

            if use_teacher_forcing:
                # Teacher forcing: Feed the target as the next input
                for di in range(target_length):
                    decoder_output, decoder_hidden, decoder_attention = decoder(
                        decoder_input, decoder_hidden, encoder_outputs, pg_mat)
                    loss += criterion(decoder_output, target_tensor[di])
                    decoder_input = target_tensor[di]  # Teacher forcing

            else:
                # Without teacher forcing: use its own predictions as the next input
                for di in range(target_length):
                    decoder_output, decoder_hidden, decoder_attention = decoder(
                        decoder_input, decoder_hidden, encoder_outputs, pg_mat)
                    topv, topi = decoder_output.topk(1)
                    decoder_input = topi.squeeze().detach()  # detach from history as input

                    loss += criterion(decoder_output, target_tensor[di])
                    if decoder_input.item() == 1:
                        break


            loss.backward()

            clipping_value = 1#arbitrary number of your choosing
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), clipping_value)
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), clipping_value)

            encoder_optimizer.step()
            decoder_optimizer.step()

            total_loss += loss.item() / target_length

        print (timeSince(start))
        print (total_loss)
        print(evaluate(encoder, decoder, test_sent[0], chi_lang, jap_lang, max_length=100))
        print(test_sent[1])





