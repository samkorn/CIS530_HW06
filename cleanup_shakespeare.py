from random import shuffle

lines = open("practical-pytorch/char-rnn-generation/shakespeare.txt", errors='ignore').read().strip().split('\n')
rand = list(range(len(lines)))
shuffle(rand)
rand = rand[:1500]
print(rand)
print(len(rand))

lines_small = [lines[idx] for idx in rand]
print(lines_small[30:40])
print(len(lines_small))

with open("shakespeare_1500.txt", "w") as f:
    for line in lines_small:
        f.write(line)
        f.write('\n')