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
    for sentence, tags in data:
        # change ' '  to '<space>'
        #sentence = map(lambda x: '<space>' if x == ' ' else x, sentence)
        #tags = map(lambda x: '<space>' if x == ' ' else x, tags)

        # remove ' '
        removed_sentence, removed_tags = [], []
        for char, tag in zip(sentence, tags):
            if char != ' ':
                removed_sentence.append(char)
                removed_tags.append(tag)
        sentence = removed_sentence
        tags = removed_tags

        sentence = ' '.join(sentence)
        tags = ' '.join(tags)
        text.append('\t'.join([sentence, tags]))

    # write
    with open(output_file, 'w') as f:
        print('\n'.join(text), file=f)
