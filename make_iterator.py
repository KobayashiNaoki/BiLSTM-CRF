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
    TAG = data.Field(sequential=True, tokenize=tokenizer, include_lengths=True, batch_first=True)
    fields = [('text', TEXT), ('tag', TAG)]
    
    train = data.TabularDataset(
        path=train_file, format='tsv',
        fields=fields)
    
    TEXT.build_vocab(train, min_freq=1)
    TAG.build_vocab(train, min_freq=1)
        
    train_iter = data.Iterator(
        train, batch_size=batch_size, device=device, repeat=False,
        sort_key=lambda x: data.interleave_keys(len(x.text)))

    """
    return: iterator, data-size, vocabs(string to index)
    """
    return train_iter, len(train), (TEXT.vocab.stoi, TAG.vocab.stoi), device, fields


def make_valid_iterator(valid_file, batch_size, device, fields):
    if type(device) == int:
        if device < 0:
            device = torch.device('cpu')
        else:
            device = torch.device('cuda:{}'.format(device))
    
    valid = data.TabularDataset(
        path=valid_file, format='tsv',
        fields=fields)
    
    valid_iter = data.Iterator(
        valid, batch_size=batch_size, device=device, repeat=False,
        sort_key=lambda x: data.interleave_keys(len(x.text)))
    
    return valid_iter, len(valid)
    

if __name__ == '__main__':
    # Usage
    BATCHSIZE = 64
    DEVICE = 1
    FILE_NAME = 'train.txt'
    EPOCH = 10
    train_iter, size, vocabs, device = make_train_iterator(FILE_NAME, BATCHSIZE, DEVICE)
    print('DATASIZE :{}'.format(size))

    print('vocab sample (TEXT)')
    for i, (word, index) in enumerate(vocabs[0].items()):
        if i > 10 :
            break
        print(word, index)
        
    print('vocab sample (TAG)')
    for i, (word, index) in enumerate(vocabs[1].items()):
        if i > 10 :
            break
        print(word, index)

    # learning loot
    for epoch in range(EPOCH): # epoch loop
        for iteration, batch in enumerate(train_iter): # iteration loop
            print(batch.text[0].size()) # shape matrix
            print(batch.text[0]) # char data
            print(batch.text[1]) # char data length
            print(batch.tag[0].size()) # shape matrix
            print(batch.tag[0])  # tag data
            print(batch.tag[1])  # tag data length
            """
            loss = model(batch.text[0], batch.tag[0]) # forward computaiton
            loss.backword() # compute backprop
            opti.step() # parameter update
            etc...
            """
            exit()
    
