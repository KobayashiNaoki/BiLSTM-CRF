import pickle
import sys

if __name__ == '__main__':
    input_file = sys.argv[1]
    output_file = sys.argv[2]

    # load
    with open(input_file, 'rb') as f:
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
    with open(output_file, 'w') as f:
        print('\n'.join(text), file=f)
