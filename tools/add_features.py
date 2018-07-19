#!/usr/bin/python3
import sys
import MeCab

def make_features(sentence, parser):
    pos_feats = []
    position_feats = []

    sentence = sentence.replace(' ', '')
    for line in parser(sentence).split('\n'):
        try:
            word, others = line.split('\t')
        except:#EOS
            break
        pos = others.split(',')[0]
        for i, char in enumerate(word):
            if len(word) == 1:
                #single character
                position_pos_tag = 'S'
            else:
                #multi characters
                if i == 0:
                    #First character
                    position_pos_tag = 'B'
                elif i == len(word)-1:
                    #last character
                    position_pos_tag = 'E'
                else:
                    #other character
                    position_pos_tag = 'I'
            pos_feats.append(pos)
            position_feats.append(position_pos_tag)
    return pos_feats, position_feats

if __name__ == '__main__':

    for line in sys.stdin:
        mecab = MeCab.Tagger()
        # mecab.parse(str)
        sentence, tags = line.rstrip().split('\t')
        pos_feat, position_feat = make_features(sentence, mecab.parse)
        pos_feat = ' '.join(pos_feat)
        position_feat = ' '.join(position_feat)
        line = '\t'.join([sentence, tags, pos_feat, position_feat])
        sys.stdout.write(line + '\n')
