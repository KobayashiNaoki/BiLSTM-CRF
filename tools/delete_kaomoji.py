import sys
for line in sys.stdin:
    
    sentence, tags, pos, position = line.strip().split('\t')
    sentence = sentence.split()
    tags = tags.split()
    pos = pos.split()
    position = position.split()
    
    if 'I' in tags or 'B' in tags:
        pair = filter(lambda pair: pair[1] == 'O', zip(sentence, tags, pos, position))
        pair = list(zip(*pair))
        if len(pair) == 0:
            continue
        sentence = pair[0]
        tags = pair[1]
        pos = pair[2]
        position = pair[3]

    sentence = ' '.join(sentence)
    tags = ' '.join(tags)
    pos = ' '.join(pos)
    position = ' '.join(position)
    
    print('\t'.join([sentence, tags, pos, position]))
