import re
author_patt = re.compile("        ― .+")

lines = open("raw_quotes.txt", errors='ignore').read().strip().split('%\n')
clean_lines = []

for line in lines:
    sub_lines = line.strip().split('\n')
    clean_sub_lines = []

    for sub_line in sub_lines:
        if author_patt.fullmatch(sub_line) is None:
            clean_sub_lines.append(sub_line)
        else:
            break

    line = ' '.join(clean_sub_lines)

    if line[0] == '"' and line[-1] == '"':
        line = line[1:-1]

    clean_lines.append(line)

with open("quotes.txt", "w") as f:
    for clean_line in clean_lines:
        f.write(clean_line)
        f.write('\n')




