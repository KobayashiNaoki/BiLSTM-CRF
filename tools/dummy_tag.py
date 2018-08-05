#!/usr/bin/python3
import sys
if __name__ == '__main__':
    for line in sys.stdin:
        sentence = ' '.join(list(line.strip()))
        if sentence == '':
            continue
        tags = ('O '*len(sentence.split())).strip()
        line = '\t'.join([sentence, tags])
        if len(sentence.split()) != len(tags.split()):
            continue
        sys.stdout.write(line + '\n')
