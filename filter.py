with open('dataset_clean.txt') as f:
    data = [line.strip() for line in f.readlines()]

text = []
for line in data:
    sentence, tag = line.split('\t')
    if len(sentence.split()) != len(tag.split()):
        continue
    else:
        text.append(line)

with open('dataset_clean.txt', 'w') as f:
    print('\n'.join(text), file=f)
