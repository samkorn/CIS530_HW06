import glob
import unicodedata
import string
import torch
import torch.nn as nn
from torch.autograd import Variable

all_letters = string.ascii_letters + " .,;'"
n_letters = len(all_letters)


# Turn a Unicode string to plain ASCII
def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )


# Read a file and split into lines
def read_lines(filename):
    lines = open(filename).read().strip().split('\n')
    return [unicode_to_ascii(line) for line in lines]


# Just for demonstration, turn a letter into a <1 x n_letters> Tensor
def letter_to_tensor(letter):
    tensor = torch.zeros(1, n_letters)
    letter_index = all_letters.find(letter)
    tensor[0][letter_index] = 1
    return tensor


# Turn a line into a <line_length x 1 x n_letters>,
# or an array of one-hot letter vectors
def line_to_tensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        letter_index = all_letters.find(letter)
        tensor[li][0][letter_index] = 1
    return tensor


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax()

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def init_hidden(self):
        return Variable(torch.zeros(1, self.hidden_size))


if __name__ == "__main__":

    all_filenames = glob.glob('data/names/*.txt')
    print(all_filenames)

    print(unicode_to_ascii('Ślusàrski'))

    # Build the category_lines dictionary, a list of names per language
    category_lines = {}
    all_categories = []

    for filename in all_filenames:
        category = filename.split('/')[-1].split('.')[0]
        all_categories.append(category)
        lines = read_lines(filename)
        category_lines[category] = lines

    n_categories = len(all_categories)
    print('n_categories =', n_categories)
    print(category_lines['Italian'][:100])

    # print(letter_to_tensor('J'))
    print(line_to_tensor('Jones').size())

    n_hidden = 128
    rnn = RNN(n_letters, n_hidden, n_categories)

    input = Variable(letter_to_tensor('A'))
    hidden = rnn.init_hidden()

    output, next_hidden = rnn(input, hidden)
    print('output.size =', output.size())

