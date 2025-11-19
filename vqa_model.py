# coding=utf-8
# Copyleft 2019 project LXRT.

import torch.nn as nn
import torch
from model.lxrt.param import args
from model.lxrt.entry import LXRTEncoder
from torch.nn.utils.weight_norm import weight_norm
import torch.nn.functional as F
import model.vqa_debias_loss_functions as vqa_loss_fc
import model.losses as lossfunc

# Max length including <bos> and <eos>
MAX_VQA_LENGTH = 16


def instance_bce_with_logits(logits, labels, reduction='mean'):
    assert logits.dim() == 2
    loss = F.binary_cross_entropy_with_logits(logits, labels, reduction=reduction)
    if reduction == "mean":
        loss *= labels.size(1)
    return loss

def compute_self_loss(logits_neg, a):
    prediction_ans_k, top_ans_ind = torch.topk(F.softmax(a, dim=-1), k=3, dim=-1, sorted=False)
    # b = F.softmax(logits_neg, dim=-1)
    neg_top_k = torch.gather(F.softmax(logits_neg, dim=-1), 1, top_ans_ind).sum(1)
    qice_loss = neg_top_k.mean()
    return qice_loss

def compute_self_loss2(logits_neg, a):
    pred_ind = torch.argsort(logits_neg, 1, descending=True)[:, :1]
    false_ans = torch.ones(logits_neg.shape[0], logits_neg.shape[1]).cuda()
    false_ans.scatter_(1, pred_ind.long(), 0)
    labels_neg = a * false_ans
    prediction_ans_k, top_ans_ind = torch.topk(F.softmax(labels_neg, dim=-1), k=1, dim=-1, sorted=False)
    # b = F.softmax(logits_neg, dim=-1)
    neg_top_k = torch.gather(F.softmax(logits_neg, dim=-1), 1, top_ans_ind) * mask
    qice_loss = neg_top_k.sum(1).mean()
    return qice_loss


class MultiDimensionalAnswerPrediction(nn.Module):
    def __init__(self, hidden_size, num_answers):
        super().__init__()
        self.hidden_size = hidden_size
        

        self.gate_fc = nn.Linear(hidden_size * 2, hidden_size)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_size * 2, num_answers)
        )

    def forward(self, q_Gen, F, v_Gen):

        F_v = F   
        gate_input = torch.cat([F_v, v_Gen], dim=-1)
        g = torch.sigmoid(self.gate_fc(gate_input))
        F_multi = g * F_v + (1 - g) * v_Gen
        logits = self.classifier(F_multi)
        return logits

class VQAModel(nn.Module):
    def __init__(self, num_answers):
        super().__init__()

        # Build LXRT encoder
        self.lxrt_encoder = LXRTEncoder(
            args,
            max_seq_length=MAX_VQA_LENGTH
        )
        hid_dim = self.lxrt_encoder.dim
        self.debias_loss_fn = vqa_loss_fc.LearnedMixin(0.36)

        # VQA Answer heads
        self.answer_predictor = MultiDimensionalAnswerPrediction(
            hidden_size=hid_dim, 
            num_answers=27294
        )

    def forward(self, feat, pos, sent, labels, bias, label_index=None, qid=None, mask=None, mode='train'):
        x, pn_loss, x_neg = self.lxrt_encoder(sent, (feat, pos), visual_attention_mask=mask, qid=qid, mode=mode)
        

        q_Gen = x.mean(dim=1) 
        v_Gen = feat.mean(dim=1) if feat.dim() == 3 else feat
        F = x 
        
        logit = self.answer_predictor(q_Gen, F, v_Gen)
        
        if labels is not None:
            loss = self.debias_loss_fn(x, logit, bias, labels)
            if pn_loss is None:
                loss_all = loss
            else:
                loss_all = pn_loss + loss
        else:
            loss_all = None
            loss = None
            pn_loss = None

        return logit, (loss_all, loss, pn_loss)


class FCNet(nn.Module):
    """Simple class for non-linear fully connect network
    """

    def __init__(self, dims, act='ReLU', dropout=0, bias=True):
        super(FCNet, self).__init__()

        layers = []
        for i in range(len(dims) - 2):
            in_dim = dims[i]
            out_dim = dims[i + 1]
            if 0 < dropout:
                layers.append(nn.Dropout(dropout))
            layers.append(weight_norm(nn.Linear(in_dim, out_dim, bias=bias),
                                      dim=None))
            if '' != act and act is not None:
                layers.append(getattr(nn, act)())
        if 0 < dropout:
            layers.append(nn.Dropout(dropout))
        layers.append(weight_norm(nn.Linear(dims[-2], dims[-1], bias=bias),
                                  dim=None))
        if '' != act and act is not None:
            layers.append(getattr(nn, act)())

        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


class SimpleClassifier(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, dropout):
        super(SimpleClassifier, self).__init__()
        layers = [
            weight_norm(nn.Linear(in_dim, hid_dim), dim=None),
            nn.ReLU(),
            # nn.GELU(),
            nn.Dropout(dropout, inplace=False),
            weight_norm(nn.Linear(hid_dim, out_dim), dim=None)
        ]
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        logits = self.main(x)
        return logits
