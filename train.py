import sys
import argparse
from pathlib import Path

from make_iterator import make_train_iterator, make_valid_iterator
from kaomoji_tagger import KaomojiTagger

import torch
import torch.nn as nn
import torch.optim as optim

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-no_CRF', action='store_true',
                        help='Flag CRF layer [default: use CRF]')
    parser.add_argument('-no_features', action='store_true',
                        help='Flag not use features  [default: use features]')
    parser.add_argument('-optimizer', type=str, choices=['SGD', 'Adam'], default='SGD',
                        help='Select Optimizer SGD or Adam [default: SGD]')
    parser.add_argument('-gpu', type=int, default=-1,
                        help='GPU ID [default: -1]')
    parser.add_argument('-epoch', type=int, default=5,
                        help='Num of Epoch [default: 5]')
    parser.add_argument('-batch_size', type=int, default=512,
                        help='Batch size [default: 512]')
    parser.add_argument('-embed_size', type=int, default=128,
                        help='Size of embedding dimension [default: 128]')
    parser.add_argument('-feat_embed_size', type=int, default=16,
                        help='Size of embedding dimension [default: 16]')
    parser.add_argument('-hidden_size', type=int, default=128,
                        help='Number of node in hidden layers [default: 128]')
    parser.add_argument('-train_file', type=str, default='data/train.txt',
                        help='Train dataset file [default: data/train.txt]')
    parser.add_argument('-valid_file', type=str, default='data/valid.txt',
                        help='Validation dataset file [default: data/valid.txt]')
    parser.add_argument('-test_file', type=str, default=None,
                        help='Test dataset file [default: None]')
    parser.add_argument('-out_dir', type=str, default='result',
                        help='Directory to Output [default: result]')
    parser.add_argument('-resume', type=str, default=None,
                        help='Resume trained model [default: None]')
    parser.add_argument('-tensorboard', action='store_true',
                        help='Logging to tensorboard (pip install tensorboardX) [default: False]')
    parser.add_argument('-test_only', action='store_true')

    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)

    # load dataset
    train_iter, train_size, vocabs, device, fields = make_train_iterator(args.train_file, args.batch_size, args.gpu)
    print('vocab size (include <unk>, <pad>)| char:{}, tag:{}, pos:{}, position:{}'.format(
        len(vocabs[0]), len(vocabs[1]), len(vocabs[2]), len(vocabs[3])), file=sys.stderr)
    O_tag_id = vocabs[1].stoi['O'] #tag vocab
    valid_iter, valid_size = make_valid_iterator(args.valid_file, args.batch_size, device, fields)
    if args.test_file is not None:
        test_iter, test_size = make_valid_iterator(args.test_file, args.batch_size, device, fields)

    # Setup a neural network
    model = KaomojiTagger(len(vocabs[0]), len(vocabs[1]),
                          [len(vocabs[2]), len(vocabs[3])],
                          args.embed_size, args.feat_embed_size, args.hidden_size,
                          dropout_p=0.1, use_CRF=not args.no_CRF, use_feature=not args.no_features)
    model.to(device=device)

    # Setup an optimizer
    if args.optimizer == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=0.1)
    elif args.optimizer == 'Adam':
        optimizer = optim.Adam(model.parameters())

    # load the snapshot
    if args.resume:
        snapshot = torch.load(args.resume)
        model.load_state_dict(snapshot['model'])
        optimizer.load_state_dict(snapshot['optimizer'])
        start_epoch = snapshot['epoch']
    else:
        start_epoch = 0

    # test_only mode
    if args.test_only:
        valid_loss, valid_accu_all, valid_accu_bi = valid_loop(valid_iter, model, O_tag_id)
        print(valid_accu_all, valid_accu_bi, file=sys.stderr)
        if args.test_file is not None:
            test_loss, test_accu_all, test_accu_bi = valid_loop(test_iter, model, O_tag_id)
            print(test_accu_all, test_accu_bi, file=sys.stderr)
            kao_dict = extract_kaomoji(test_iter, model, O_tag_id, vocabs[0])
            print('\n'.join(kao_dict))
        return 0

    # prepare a tensorboard
    if args.tensorboard:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter(log_dir=out_dir)

    # train model
    for epoch in range(start_epoch, args.epoch):
        # train step
        train_loss = train_loop(train_iter, model, optimizer)
        # validation step
        valid_loss, valid_accu_all, valid_accu_bi = valid_loop(valid_iter, model, O_tag_id)
        # save snapshot
        torch.save(
            {'epoch': epoch,
             'train/loss': train_loss,
             'valid/loss': valid_loss,
             'valid/accu': valid_accu_all,
             'model': model.state_dict(),
             'optimizer': optimizer.state_dict()},
            out_dir / 'model_{}.pt'.format(epoch))
        # logging
        print('epoch:{}, train/loss:{}, valid/loss:{}, valid/accu(ALL):{}, valid/accu(BI):{}'.format(
            epoch, train_loss, valid_loss, valid_accu_all, valid_accu_bi), file=sys.stderr)
        if args.tensorboard:
            writer.add_scalar('train/loss', train_loss, epoch)
            writer.add_scalar('valid/loss', valid_loss, epoch)
            writer.add_scalar('valid/accu/all', valid_accu_all, epoch)
            writer.add_scalar('valid/accu/bi', valid_accu_bi, epoch)

    if args.tensorboard:
        writer.export_scalars_to_json(out_dir / "log.json")
        writer.close()

    return 0

def train_loop(train_iter, model, optimizer):
    model.train()
    sum_loss = 0
    normalize = 0
    for iteration, batch in enumerate(train_iter):
        src = batch.text[0]
        trg = batch.tag
        feats = (batch.pos, batch.position)
        length = batch.text[1]
        # forward computation
        loss = model(src, trg, feats, length)
        m_loss = loss.mean()
        # init gradient
        optimizer.zero_grad()
        # backward computation
        m_loss.backward()
        # update gradient
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        sum_loss += loss.sum().item()
        normalize += length.size()[0]
    avg_loss = sum_loss / normalize
    return avg_loss

def valid_loop(valid_iter, model, O_tag_id):
    model.eval()
    sum_loss, sum_accu_all, sum_accu_bi = 0, 0, 0
    normalize = 0
    for iteration, batch in enumerate(valid_iter):
        src = batch.text[0]
        trg = batch.tag
        feats = (batch.pos, batch.position)
        length = batch.text[1]
        # forward computation
        loss = model(src, trg, feats, length)
        sum_loss += loss.sum().item()
        # label prediction
        label, scores = model.predict(src, feats, length, return_scores=True)
        normalize += length.size()[0]
        # calc accuracy (all tag)
        target_tag = ((trg != 1)) #1:padding
        correct = ((trg == label) & target_tag).sum(dim=1) #dim=1: sum per sentences
        accu = correct.float() / target_tag.sum(dim=1).float()
        sum_accu_all += accu.sum().item()
        # calc accuracy (bi tag)
        target_tag = ((trg != 1) & (trg != O_tag_id)) #1:padding, 2:O tag
        correct = ((trg == label) & target_tag).sum(dim=1) #dim=1: sum per sentences
        accu = correct.float() / target_tag.sum(dim=1).float()
        sum_accu_bi += accu.sum().item()
    # normalize
    avg_loss = sum_loss / normalize
    avg_accu_all = sum_accu_all / normalize
    avg_accu_bi = sum_accu_bi / normalize
    model.train()
    return avg_loss, avg_accu_all, avg_accu_bi

def extract_kaomoji(test_iter, model, O_tag_id, vocab):
    kaomoji_dict = []
    model.eval()
    for iteration, batch in enumerate(test_iter):
        src = batch.text[0]
        feats = (batch.pos, batch.position)
        length = batch.text[1]
        # label prediction
        label, scores = model.predict(src, feats, length, return_scores=True)
        kaomoji_dict.extend(_extract_kaomoji(label, src, O_tag_id, vocab))
    return kaomoji_dict

def _extract_kaomoji(tags, sentences, O_tag_id, vocab):
    tags = tags.cpu().numpy()
    sentences = sentences.cpu().numpy()
    kaomojis = []
    for labels, sentence in zip(tags, sentences): #batch loop
        kaomoji = []
        for label, char_id in zip(labels, sentence):
            char = vocab.itos[char_id]
            if char == '<pad>':
                break
            if label != O_tag_id:
                kaomoji.append(char)
            elif label == O_tag_id and len(kaomoji) > 0:
                kaomojis.append(''.join(kaomoji))
                kaomoji = []
        if len(kaomoji) > 0:
            kaomojis.append(''.join(kaomoji))
            kaomoji = []
    return kaomojis

if __name__ == '__main__':
    main()
