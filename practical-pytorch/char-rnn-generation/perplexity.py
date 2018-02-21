import argparse
import torch
from helpers import *
from model import *
<<<<<<< HEAD
import numpy as np
=======
>>>>>>> master

def evaluate(line_tensor):
    hidden = decoder.init_hidden()
    for i in range(line_tensor.size()[0]):
<<<<<<< HEAD
        output, hidden = decoder(line_tensor[i], hidden)
=======
        output, hidden = rnn(line_tensor[i], hidden)
>>>>>>> master

    return output

def perplexity(decoder, test_filename):
<<<<<<< HEAD
    softmax = torch.nn.LogSoftmax(dim=1)
=======

>>>>>>> master
    """

    :param decoder:
    :param test_filename:
    :type decoder: torch.nn.Module
    :return:
    """
    test = open(test_filename, encoding='iso-8859-1').read()
<<<<<<< HEAD
=======
    char_tensor(
>>>>>>> master
    #pad = "~" * order
    #test = pad + test

    sz = len(test)
    log_sum = 0
<<<<<<< HEAD

    curr_char = test[0]
    for i in range(sz - 1):

        curr_char_tensor = char_tensor(str(curr_char))
        output = softmax(evaluate(curr_char_tensor))

        next_char = test[i + 1]
        index_next_char = all_characters.index(string[next_char])
        prob = output[index_next_char]

        """next_letter_prob = model["<UNK>"]
=======
    curr_char = test[0]
    for i in range(sz):
        output = evaluate()
        next_letter = test[i + order]
        next_letter_prob = model["<UNK>"]
>>>>>>> master
        if phrase in model:
            phrase_probs = model[phrase]
            for j in range(len(phrase_probs)):
                tup = phrase_probs[j]
                if tup[0] == next_letter:
                    next_letter_prob = tup[1]
<<<<<<< HEAD
                    break"""
        log_sum += np.log(prob)
        curr_char = test[i + 1]
=======
                    break
        log_sum += np.log(next_letter_prob)
>>>>>>> master
    return log_sum * (-1 / float(sz))



if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('model_filename', type=str)
    argparser.add_argument('test_filename', type=str)
    args = argparser.parse_args()
    decoder = torch.load(args.filename)
    print(perplexity(decoder, **vars(args)))

