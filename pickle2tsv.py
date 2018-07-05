import pickle

# load
with open('dataset.pkl', 'rb') as f:
    data = pickle.load(f)

# convert
text = []
for sentence, tag in data:
    # change ' '  to '<space>'
    sentence = map(lambda x: '<space>' if x == ' ' else x, sentence)
    tag = map(lambda x: '<space>' if x == ' ' else x, tag)
    sentence = ' '.join(sentence)
    tag = ' '.join(tag)
    text.append('\t'.join([sentence, tag]))

# write
with open('dataset.txt', 'w') as f:
    print('\n'.join(text), file=f)
