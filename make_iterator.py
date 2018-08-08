from torchtext import data
import torch
import unicodedata


def make_train_iterator(train_file, batch_size, device):
    """
    Arguments: batch_size, device
    """
    
    if device < 0:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:{}'.format(device))
        
    def tokenizer(sentence):
        sentence = unicodedata.normalize("NFKC", sentence)
        return sentence.split()
    
    TEXT = data.Field(sequential=True, tokenize=tokenizer, include_lengths=True, batch_first=True)
    TAG = data.Field(sequential=True, tokenize=tokenizer, batch_first=True)
    POS = data.Field(sequential=True, tokenize=tokenizer, batch_first=True)
    POSITION = data.Field(sequential=True, tokenize=tokenizer, batch_first=True)
    fields = [('text', TEXT), ('tag', TAG), ('pos', POS), ('position', POSITION)]
    
    train = data.TabularDataset(
        path=train_file, format='tsv',
        fields=fields)

    dataset_filter(train)
    
    TEXT.build_vocab(train, min_freq=2)
    TAG.build_vocab(train, min_freq=1)
    POS.build_vocab(train, min_freq=1)
    POSITION.build_vocab(train, min_freq=1)
    
    train_iter = data.Iterator(
        train, batch_size=batch_size, device=device, repeat=False,
        sort_key=lambda x: data.interleave_keys(len(x.text)))

    """
    return: iterator, data-size, vocabs(string to index)
    """
    return train_iter, train, len(train), (TEXT.vocab, TAG.vocab, POS.vocab, POSITION.vocab), device, fields


def make_valid_iterator(valid_file, batch_size, device, fields):
    if type(device) == int:
        if device < 0:
            device = torch.device('cpu')
        else:
            device = torch.device('cuda:{}'.format(device))
    
    valid = data.TabularDataset(
        path=valid_file, format='tsv',
        fields=fields)

    dataset_filter(valid)

    valid_iter = data.Iterator(
        valid, batch_size=batch_size, device=device, repeat=False, sort=False, shuffle=False)
    
    return valid_iter, valid, len(valid)
    

def dataset_filter(dataset):
    examples = []
    for d in dataset.examples:
        if len(d.text) == len(d.tag):
            examples.append(d)
    dataset.examples = examples


if __name__ == '__main__':
    # Usage
    BATCHSIZE = 64
    DEVICE = 0
    FILE_NAME = 'data/test.txt'
    EPOCH = 10
    train_iter, size, vocabs, device, fields = make_train_iterator(FILE_NAME, BATCHSIZE, DEVICE)
    print('DATASIZE :{}'.format(size))

    print('vocab sample (TEXT)')
    for i, (word, index) in enumerate(vocabs[0].stoi.items()):
        if i > 10 :
            break
        print(word, index)
        
    print('vocab sample (TAG)')
    for i, (word, index) in enumerate(vocabs[1].stoi.items()):
        if i > 10 :
            break
        print(word, index)

    print('vocab sample (POS)')
    for i, (word, index) in enumerate(vocabs[2].stoi.items()):
        if i > 10 :
            break
        print(word, index)

    print('vocab sample (POSITION)')
    for i, (word, index) in enumerate(vocabs[3].stoi.items()):
        if i > 10 :
            break
        print(word, index)

    # learning loot
    for epoch in range(EPOCH): # epoch loop
        for iteration, batch in enumerate(train_iter): # iteration loop
            if batch.text[0].size() != batch.tag.size():
                print('error')
                print(batch.text[0].size()) # shape matrix
                print(batch.text[0]) # char data
                print(batch.text[1]) # char data length
                print(batch.tag.size()) # shape matrix
                print(batch.tag)  # tag data
                print(batch.pos.size()) # shape matrix
                print(batch.pos)  # pos data
                print(batch.position.size()) # shape matrix
                print(batch.position)  # position data
            else:
                continue
            """
            loss = model(batch.text[0], batch.tag[0]) # forward computaiton
            loss.backword() # compute backprop
            opti.step() # parameter update
            etc...
            """
    
