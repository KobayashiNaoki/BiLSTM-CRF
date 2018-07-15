#!/usr/bin/python3
import sys
if __name__ == '__main__':

    for line in sys.stdin:
        sentence, tag = line.split('\t')
        if len(sentence.split()) != len(tag.split()):
            continue
        sys.stdout.write(line)
