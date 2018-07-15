#!/usr/bin/python3
import sys
def abio2bio(tag):
    if tag == 'A': return 'O'
    elif tag == 'O': return 'I'
    else : return tag

if __name__ == '__main__':

    for line in sys.stdin:
        sentence, tags = line.split('\t')
        tags = ' '.join(map(abio2bio, tags.split()))
        line = '\t'.join([sentence, tags])
        sys.stdout.write(line + '\n')
