import glob
import unicodedata
import string
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

import random
import time
import math

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


output_labels = []
all_letters = string.ascii_letters + " .,;'"
n_letters = len(all_letters)


category_lines_train = {}
category_lines_val = {}
all_categories = []
train_filenames = glob.glob('data/cities_train/*.txt')
#train_filenames = glob.glob('data/cities_train/*.txt')
val_filenames = glob.glob('data/cities_val/*.txt')

# Turn a Unicode string to plain ASCII
def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )


# Read a file and split into lines
def readLines(filename):
    lines = open(filename, errors='ignore').read().strip().split('\n')
    return [unicode_to_ascii(line) for line in lines]

for filename in train_filenames:
    category = filename.split('/')[-1].split('.')[0]
    all_categories.append(category)
    lines = readLines(filename)
    category_lines_train[category] = lines

for filename in val_filenames:
    category = filename.split('/')[-1].split('.')[0]
    lines = readLines(filename)
    category_lines_val[category] = lines

n_categories = len(all_categories)
print('n_categories =', n_categories)

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


def category_from_output(output):
    top_n, top_i = output.data.topk(1)  # Tensor out of Variable with .data
    category_i = top_i[0][0]
    return all_categories[category_i], category_i


def random_training_pair(all_categories):
    category = random.choice(all_categories)
    line = random.choice(category_lines_train[category])
    category_tensor = Variable(torch.LongTensor([all_categories.index(category)]))
    line_tensor = Variable(line_to_tensor(line))
    return category, line, category_tensor, line_tensor


def train(category_tensor, line_tensor):
    # If you set this too high, it might explode.
    # If too low, it might not learn
    rnn.zero_grad()
    hidden = rnn.init_hidden()

    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)

    loss = criterion(output, category_tensor)
    loss.backward()

    optimizer.step()

    return output, loss.data[0]


def time_since(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


# Just return an output given a line
def evaluate(line_tensor):
    hidden = rnn.init_hidden()

    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)

    return output


def predict(input_line, n_predictions=3):
    print('\n> %s' % input_line)
    output = evaluate(Variable(line_to_tensor(input_line)))

    # Get top N categories
    topv, topi = output.data.topk(n_predictions, 1, True)
    predictions = []

    for i in range(n_predictions):
        value = topv[0][i]
        category_index = topi[0][i]
        print('(%.2f) %s' % (value, all_categories[category_index]))
        predictions.append([value, all_categories[category_index]])


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def init_hidden(self):
        return Variable(torch.zeros(1, self.hidden_size))


def compute_validation_loss():
    counter = 0
    validation_loss = 0
    for category in all_categories:
        validation_category_tensor = Variable(torch.LongTensor([all_categories.index(category)]))
        for line in category_lines_val[category]:
            validation_line_tensor = Variable(line_to_tensor(line))
            out = evaluate(validation_line_tensor)
            validation_loss += criterion(out, validation_category_tensor).data[0]
            counter += 1
    validation_loss = validation_loss / counter
    return validation_loss



# Keep track of correct guesses in a confusion matrix
"""confusion = torch.zeros(n_categories, n_categories)
    n_confusion = 10000
    # Go through a bunch of examples and record which are correctly guessed
    for i in range(n_confusion):
        category, line, category_tensor, line_tensor = random_training_pair(all_categories)
        output = evaluate(line_tensor)
        guess, guess_i = category_from_output(output)
        category_i = all_categories.index(category)
        confusion[category_i][guess_i] += 1

    # Normalize by dividing every row by its sum
    for i in range(n_categories):
        confusion[i] = confusion[i] / confusion[i].sum()

    # Set up plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(confusion.numpy())
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + all_categories, rotation=90)
    ax.set_yticklabels([''] + all_categories)

    # Force label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    plt.show()"""

#############################


# [1]

# print(all_filenames)

# print(unicode_to_ascii('Ślusàrski'))

# Build the category_lines dictionary, a list of names per language
# [3]


n_categories = len(all_categories)

# print('n_categories =', n_categories)

# [4]
# print(category_lines['de'][:100])

# [6]
# print(letter_to_tensor('t'))
# [7]
# print(line_to_tensor('torce').size())

# MODIFY THIS VV ???
n_hidden = 128
rnn = RNN(n_letters, n_hidden, n_categories)

input = Variable(line_to_tensor('torce'))
hidden = rnn.init_hidden()

output, next_hidden = rnn.forward(input[0], hidden)

# print('output.size =', output.size())
# print(output)
# print(category_from_output(output))

# for i in range(10):
#     category, line, category_tensor, line_tensor = random_training_pair(all_categories)
#     print('category =', category, '/ line =', line)
lrs = np.arange(0.0001, 0.002, .0002)

for learning_rate in lrs:
    criterion = nn.NLLLoss()
    #learning_rate = 0.001
    optimizer = torch.optim.SGD(rnn.parameters(), lr=learning_rate)

    n_epochs = 100000
    print_every = 5000
    plot_every = 1000

    # Keep track of losses for plotting
    current_train_loss = 0
    train_losses = []
    validation_losses = []
    start = time.time()
    for epoch in range(1, n_epochs + 1):

        # Get a random training input and target
        category, line, category_tensor, line_tensor = random_training_pair(all_categories)
        output, loss = train(category_tensor, line_tensor)
        current_train_loss += loss

        # Print epoch number, loss, name and guess
        if epoch % print_every == 0:
            guess, guess_i = category_from_output(output)
            correct = '✓' if guess == category else '✗ (%s)' % category

            print('%d %d%% (%s) %.4f %s / %s %s' % (
                epoch, epoch / n_epochs * 100, time_since(start), loss, line, guess,
                correct))

        # Add current loss avg to list of losses
        if epoch % plot_every == 0:
            train_losses.append(current_train_loss / plot_every)
            validation_losses.append(compute_validation_loss())
            current_train_loss = 0

    plt.figure()
    print("training losses = " + str(train_losses))
    print("val losses = " + str(validation_losses))

    plt.plot(train_losses)
    plt.plot(validation_losses)
    plt.savefig("output/losses" + str(learning_rate) + ".png")
    #compute_validation_accuracy()


    confusion = torch.zeros(n_categories, n_categories)
    # n_confusion = 10000
    #  Go through a bunch of examples and record which are correctly guessed
    for category in all_categories:
        validation_category_tensor = Variable(torch.LongTensor([all_categories.index(category)]))
        for line in category_lines_val[category]:
            validation_line_tensor = Variable(line_to_tensor(line))
            output = evaluate(validation_line_tensor)
            guess, guess_i = category_from_output(output)
            category_i = all_categories.index(category)
            confusion[category_i][guess_i] += 1
            output_labels.append(guess)
    # Normalize by dividing every row by its sum
    for i in range(n_categories):
        confusion[i] = confusion[i] / confusion[i].sum()

    # Set up plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(confusion.numpy())
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + all_categories, rotation=90)
    ax.set_yticklabels([''] + all_categories)

    # Force label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    plt.savefig('output/validation_confusion' + str(learning_rate) + '.png')

with open('labels.txt') as labelstxt:
    for ol in output_labels:
        labelstxt.write(ol)
        labelstxt.write('\n')
    labelstxt.close()

"""for category in all_categories:
for line in category_lines_train[category]:
    category_tensor = Variable(torch.LongTensor([all_categories.index(category)]))
    line_tensor = Variable(line_to_tensor(line))
    output, loss = train(category_tensor, line_tensor)
    current_loss += loss
    count = count + 1
    if count % print_every == 0:
        guess, guess_i = category_from_output(output)
        correct = '✓' if guess == category else '✗ (%s)' % category
        print(time_since(start), loss, line, guess, correct, category)
    if count % plot_every == 0:
        # print(current_loss)
        all_losses.append(current_loss / plot_every)
        current_loss = 0
"""




# predict('villedieu')  # fr
# predict('odemis')  # de
# predict('madilo')  # za
