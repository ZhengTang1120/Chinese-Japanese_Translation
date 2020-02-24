from model import *
from languages import *
import random

def split_sentence(sentence, name):
    if name == "japanese":
        return mecab.parse(sentence).split()
    elif name == "chinese":
        return seg.cut(sentence)
    else:
        return sentence.split()

def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence]


def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(1)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)

def evaluate(encoder, decoder, sentence, input_lang, output_lang, max_length=100):
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, sentence)
        input_length = input_tensor.size()[0]

        encoder_output, encoder_hidden = encoder(input_tensor)
        encoder_outputs  = encoder_output.view(input_length, -1)

        decoder_input = torch.tensor([[0]], device=device)  # SOS

        decoder_hidden = encoder_hidden

        decoded_words = []

        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == 1:
                decoded_words.append('<EOS>')
                break
            else:
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
            if c[i].startswith('我') and len(c[i]) < 100 and n<1000:
                n+=1
                print (i)
                print (c[i])
                print (j[i])
                pairs.append((chi_lang.addSentence(c[i]), jap_lang.addSentence(j[i])))

    print (jap_lang.n_words)
    print (chi_lang.n_words)

    training_set = list()

    for pair in pairs:
        chi_sent = pair[0]
        chi_sent.reverse()
        jap_sent = pair[1]
        chi_tensor = tensorFromSentence(chi_lang, chi_sent)
        jap_tensor = tensorFromSentence(jap_lang, jap_sent)
        training_set.append((chi_tensor, jap_tensor))

    learning_rate = 0.01
    hidden_size = 256

    encoder    = EncoderRNN(chi_lang.n_words, hidden_size).to(device)
    decoder    = AttnDecoderRNN(hidden_size, jap_lang.n_words, dropout_p=0.1).to(device)

    encoder_optimizer    = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer    = optim.SGD(decoder.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()

    teacher_forcing_ratio = 0.5

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    test_sent = pairs[0]
    print(test_sent[0])
    print(test_sent[1])
    for epoch in range(20):

        random.shuffle(training_set)
        total_loss = 0

        for input_tensor, target_tensor in training_set:
            loss = 0

            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()

            input_length = input_tensor.size(0)
            target_length = target_tensor.size(0)

            encoder_output, encoder_hidden = encoder(input_tensor)
            encoder_outputs  = encoder_output.view(input_length, -1)

            decoder_input = torch.tensor([[0]], device=device)
            decoder_hidden = encoder_hidden

            use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

            if use_teacher_forcing:
                # Teacher forcing: Feed the target as the next input
                for di in range(target_length):
                    decoder_output, decoder_hidden, decoder_attention = decoder(
                        decoder_input, decoder_hidden, encoder_outputs)
                    loss += criterion(decoder_output, target_tensor[di])
                    decoder_input = target_tensor[di]  # Teacher forcing

            else:
                # Without teacher forcing: use its own predictions as the next input
                for di in range(target_length):
                    decoder_output, decoder_hidden, decoder_attention = decoder(
                        decoder_input, decoder_hidden, encoder_outputs)
                    topv, topi = decoder_output.topk(1)
                    decoder_input = topi.squeeze().detach()  # detach from history as input

                    loss += criterion(decoder_output, target_tensor[di])
                    if decoder_input.item() == 1:
                        break

            loss.backward()

            encoder_optimizer.step()
            decoder_optimizer.step()

            total_loss += loss.item() / target_length

        print (total_loss)
        print(evaluate(encoder, decoder, test_sent[0], chi_lang, jap_lang, max_length=100))
        print(test_sent[1])





