import torch
import torch.nn as nn
import torch.nn.init as I
import torch.nn.functional as F
import numpy

class KaomojiTagger(nn.Module):
    def __init__(self, source_vocab_size, target_vocab_size, feat_vocab_sizes,
                 embed_size, feat_embed_size, hidden_size,
                 dropout_p=0.1, use_CRF=True, use_feature=True):
        super(KaomojiTagger, self).__init__()
        dropout_p = dropout_p

        if use_CRF:
            self.crf = CRF(target_vocab_size)
            target_vocab_size = self.crf.n_labels
        else:
            self.loss_fun = nn.CrossEntropyLoss(ignore_index=1, size_average=False)
            self.crf = None

        self.embedding = Embedding(source_vocab_size, feat_vocab_sizes,
                                   embed_size, feat_embed_size, use_feature=use_feature)
        if use_feature:
            embed_size = embed_size + (feat_embed_size * len(feat_vocab_sizes))
        self.bilstm = BiLSTM(target_vocab_size, embed_size, hidden_size, dropout_p)

    def forward(self, src, trg, feats, length, return_logits=False):
        ex = self.embedding(src, feats)
        logits = self.bilstm(ex, length)
        if self.crf is None:
            b, l, v = logits.size()
            output = logits.view(b * l, v)
            trg = trg.view(b * l)
            loglik = -self.loss_fun(output, trg)
        else:
            norm_score = self.crf(logits, length)
            sequence_score = self.score(src, trg, length, logits=logits)
            loglik = sequence_score - norm_score

        if return_logits:
            return -loglik, logits
        else:
            return -loglik

    def predict(self, src, feats, length, return_scores=False):
        ex = self.embedding(src, feats)
        logits = self.bilstm(ex, length)
        if self.crf is None:
            preds = torch.argmax(logits, dim=2)
            scores = None
        else:
            scores, preds = self.crf.viterbi_decode(logits, length)

        if return_scores:
            return preds, scores
        else:
            return preds

    def _bilstm_score(self, logits, y, length):
        y_exp = y.unsqueeze(-1)
        scores = torch.gather(logits, 2, y_exp).squeeze(-1)
        mask = sequence_mask(length).float()
        scores = scores * mask
        score = scores.sum(1).squeeze(-1)

        return score

    def score(self, xs, y, length, logits=None):
        if logits is None:
            logits = self.bilstm(xs, length)

        transition_score = self.crf.transition_score(y, length)
        bilstm_score = self._bilstm_score(logits, y, length)

        score = transition_score + bilstm_score

        return score


class Embedding(nn.Module):
    def __init__(self, source_vocab_size, feat_vocab_sizes,
                 embed_size, feat_embed_size, dropout_p=0.1, use_feature=True):
        super(Embedding, self).__init__()
        self.char_embed = nn.Embedding(source_vocab_size, embed_size, padding_idx=1)
        self.use_feature = use_feature
        if use_feature:
            self.feats_embed = nn.ModuleList()
            for feat_size in feat_vocab_sizes:
                self.feats_embed.append(nn.Embedding(feat_size, feat_embed_size, padding_idx=1))
        self.embed_dropout = nn.Dropout(p=dropout_p)

    def forward(self, src, feats):
        char = src
        # Embedding: ex = E(x)
        ex = self.char_embed(char)  # BxL -> BxLxH
        if self.use_feature:
            embeddings = [ex]
            for feat, embed in zip(feats, self.feats_embed):
                embeddings.append(embed(feat))
            ex = torch.cat(embeddings, dim=2)
        ex = self.embed_dropout(ex)
        return ex


class BiLSTM(nn.Module):
    def __init__(self, target_vocab_size, embed_size, hidden_size, dropout_p=0.1):
        super(BiLSTM, self).__init__()
        output_size = target_vocab_size
        dropout_p = dropout_p
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers=2,
                            dropout=dropout_p, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(hidden_size * 2, output_size)

    def forward(self, ex, length):

        # Sort in decreasing order
        length, parm_idx = torch.sort(length, 0, descending=True)
        device = parm_idx.device
        parm_idx_rev = torch.tensor(_inverse_indices(parm_idx), device=device)

        # PackPadding
        packed_input = nn.utils.rnn.pack_padded_sequence(ex, length, batch_first=True)

        # LSTM: h_{t+1} = LSTM(h_{t}, x{t})
        packed_output, _ = self.lstm(packed_input)

        # UnPackPadding
        h = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        h = h[0][parm_idx_rev, :]  # h[0]:lstm output, h[1]:length

        # Linear: y = W(h)
        y = self.linear(h) # BxLx2*H -> BxLxV
        return y


def _inverse_indices(indices):
    indices = indices.cpu().numpy()
    r = numpy.empty_like(indices)
    r[indices] = numpy.arange(len(indices))
    return r


class CRF(nn.Module):
    def __init__(self, vocab_size):
        super(CRF, self).__init__()

        self.vocab_size = vocab_size
        self.n_labels = n_labels = vocab_size + 2
        self.start_idx = n_labels - 2
        self.stop_idx = n_labels - 1
        self.transitions = nn.Parameter(torch.randn(n_labels, n_labels))

    def reset_parameters(self):
        I.normal(self.transitions.data, 0, 1)

    def forward(self, logits, length):
        """
        Arguments:
            logits: [batch_size, seq_len, n_labels]
            length: [batch_size]
        """
        batch_size, seq_len, n_labels = logits.size()
        alpha = logits.data.new(batch_size, self.n_labels).fill_(-10000)
        alpha[:, self.start_idx] = 0
        alpha = torch.tensor(alpha)
        c_length = length.clone()

        logits_t = logits.transpose(1, 0)
        for logit in logits_t:
            logit_exp = logit.unsqueeze(-1).expand(batch_size,
                                                   *self.transitions.size())
            alpha_exp = alpha.unsqueeze(1).expand(batch_size,
                                                  *self.transitions.size())
            
            trans_exp = self.transitions.unsqueeze(0).expand_as(alpha_exp)
            mat = trans_exp + alpha_exp + logit_exp
            alpha_nxt = log_sum_exp(mat, 2).squeeze(-1)

            mask = (c_length > 0).float().unsqueeze(-1).expand_as(alpha)
            alpha = mask * alpha_nxt + (1 - mask) * alpha
            c_length = c_length - 1

        alpha = alpha + self.transitions[self.stop_idx].unsqueeze(0).expand_as(alpha)
        norm = log_sum_exp(alpha, 1).squeeze(-1)

        return norm

    def viterbi_decode(self, logits, length):
        """Borrowed from pytorch tutorial

        Arguments:
            logits: [batch_size, seq_len, n_labels] FloatTensor
            length: [batch_size] LongTensor
        """
        batch_size, seq_len, n_labels = logits.size()
        vit = logits.data.new(batch_size, self.n_labels).fill_(-10000)
        vit[:, self.start_idx] = 0
        vit = torch.tensor(vit)
        c_length = length.clone()

        logits_t = logits.transpose(1, 0)
        pointers = []
        for logit in logits_t:
            vit_exp = vit.unsqueeze(1).expand(batch_size, n_labels, n_labels)
            trn_exp = self.transitions.unsqueeze(0).expand_as(vit_exp)
            vit_trn_sum = vit_exp + trn_exp
            vt_max, vt_argmax = vit_trn_sum.max(2)

            vt_max = vt_max.squeeze(-1)
            vit_nxt = vt_max + logit
            pointers.append(vt_argmax.squeeze(-1).unsqueeze(0))

            mask = (c_length > 0).float().unsqueeze(-1).expand_as(vit_nxt)
            vit = mask * vit_nxt + (1 - mask) * vit

            mask = (c_length == 1).float().unsqueeze(-1).expand_as(vit_nxt)
            vit += mask * self.transitions[ self.stop_idx ].unsqueeze(0).expand_as(vit_nxt)

            c_length = c_length - 1

        pointers = torch.cat(pointers)
        scores, idx = vit.max(1)
        idx = idx.squeeze(-1)
        paths = [idx.unsqueeze(1)]

        #for argmax in reversed(pointers):
        for argmax in flip(pointers, 0):
            idx_exp = idx.unsqueeze(-1)
            idx = torch.gather(argmax, 1, idx_exp)
            idx = idx.squeeze(-1)

            paths.insert(0, idx.unsqueeze(1))

        paths = torch.cat(paths[1:], 1)
        scores = scores.squeeze(-1)

        return scores, paths

    def transition_score(self, labels, length):
        """
        Arguments:
             labels: [batch_size,s eq_len]
             length: [batch_size]
        """
        batch_size, seq_len = labels.size()

        # pad labels with <start> and <stop> indices
        labels_ext = torch.tensor(labels.data.new(batch_size, seq_len + 2))
        labels_ext[:, 0] = self.start_idx
        labels_ext[:, 1:-1] = labels
        mask = sequence_mask(length + 1, max_len=seq_len + 2).long()
        pad_stop = torch.tensor(labels.data.new(1).fill_(self.stop_idx))
        pad_stop = pad_stop.unsqueeze(-1).expand(batch_size, seq_len + 2)
        labels_ext = (1 - mask) * pad_stop + mask * labels_ext
        labels = labels_ext

        trn = self.transitions

        # obtain transition vector for each label in batch and timestep
        # (except the last ones)
        trn_exp = trn.unsqueeze(0).expand(batch_size, *trn.size())
        lbl_r = labels[:, 1:]
        lbl_rexp = lbl_r.unsqueeze(-1).expand(*lbl_r.size(), trn.size(0))
        trn_row = torch.gather(trn_exp, 1, lbl_rexp)

        # obtain transition score from the transition vector for each label
        # in batch and timestep (except the first ones)
        lbl_lexp = labels[:, :-1].unsqueeze(-1)
        trn_scr = torch.gather(trn_row, 2, lbl_lexp)
        trn_scr = trn_scr.squeeze(-1)

        mask = sequence_mask(length + 1).float()
        trn_scr = trn_scr * mask
        score = trn_scr.sum(1).squeeze(-1)

        return score
    
def log_sum_exp(vec, dim=0):
    max, idx = torch.max(vec, dim)
    max_exp = max.unsqueeze(-1).expand_as(vec)
    return max + torch.log(torch.sum(torch.exp(vec - max_exp), dim))

def sequence_mask(length, max_len=None):
    batch_size = length.size(0)

    if max_len is None:
        #max_len = length.max().data[0]
        max_len = length.max().item()

    ranges = torch.arange(0, max_len).long()
    ranges = ranges.unsqueeze(0).expand(batch_size, max_len)
    ranges = torch.tensor(ranges)

    if length.data.is_cuda:
        ranges = ranges.cuda()

    length_exp = length.unsqueeze(1).expand_as(ranges)
    mask = ranges < length_exp

    return mask

def flip(x, dim):
    indices = [slice(None)] * x.dim()
    indices[dim] = torch.arange(x.size(dim) - 1, -1, -1,
                                dtype=torch.long, device=x.device)
    return x[tuple(indices)]
