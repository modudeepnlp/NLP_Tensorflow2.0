import pickle
import argparse
import torch
import torch.nn as nn

from pathlib import Path
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
# from torch.utils.tensorboard import SummaryWriter
from model.utils import split_to_jamo
from model.data import Corpus, Tokenizer
from model.net import EfficientCharCRNN
from torch.nn.utils import clip_grad_norm_
from gluonnlp.data import PadSequence
from tqdm import tqdm
from model.metric import evaluate, acc
# from build_preprocessing import Preprocessing
from build_vocab import Build_Vocab


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', default=10, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    # parser.add_argument('--data_type', default='senCNN')
    parser.add_argument('--classes', default=2, type=int)
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--learning_rate', default=1e-3, type=float)
    # parser.add_argument('--print_freq', default=3000, type=int)
    # parser.add_argument('--weight_decay', default=5e-5, type=float)
    parser.add_argument('--word_dim', default=16, type=int)
    parser.add_argument('--word_max_len', default=300, type=int)
    parser.add_argument('--global_step', default=1000, type=int)
    parser.add_argument('--data_path', default='../data_in')
    parser.add_argument('--file_path', default='../nsmc-master')
    # parser.add_argument('--build_preprocessing', default=False)
    # parser.add_argument('--build_vocab', default=False)

    args = parser.parse_args()
    # p = Preprocessing(args)
    # p.makeProcessing()

    # v = Build_Vocab(args)
    # v.make_vocab()

    with open(args.data_path + '/' + 'vocab_char.pkl', mode='rb') as io:
        vocab = pickle.load(io)

    padder = PadSequence(length=args.word_max_len, pad_val=vocab.to_indices(vocab.padding_token))
    tokenizer = Tokenizer(vocab=vocab, split_fn=split_to_jamo, pad_fn=padder)

    model = EfficientCharCRNN(args, vocab)

    epochs = args.epoch
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    global_step = args.global_step

    tr_ds = Corpus(args.data_path + '/train.txt', tokenizer.split_and_transform)
    tr_dl = DataLoader(tr_ds, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
    val_ds = Corpus(args.data_path + '/val.txt', tokenizer.split_and_transform)
    val_dl = DataLoader(val_ds, batch_size=batch_size, num_workers=4)

    loss_fn = nn.CrossEntropyLoss()
    opt = optim.Adam(params=model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(opt, patience=5)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    best_val_loss = 1e+10

    for epoch in tqdm(range(args.epoch), desc='epochs'):
        tr_loss = 0
        tr_acc = 0
        model.train()
        for step, mb in tqdm(enumerate(tr_dl), desc='steps', total=len(tr_dl)):
            x, y = map(lambda elm:elm.to(device), mb)
            opt.zero_grad()
            y_h = model(x)
            m_loss = loss_fn(y_h, y)
            m_loss.backward()
            clip_grad_norm_(model._fc.weight, 5)
            opt.step()

            with torch.no_grad():
                m_acc = acc(y_h,y)

            tr_loss += m_loss.item()
            tr_acc += m_acc.item()

        else:
            tr_loss /= (step + 1)
            tr_acc /= (step + 1)

            tr_summ = {'loss': tr_loss, 'acc': tr_acc}
            val_summ = evaluate(model, val_dl, {'loss': loss_fn, 'acc': acc}, device)
            scheduler.step(val_summ['loss'])
            tqdm.write('epoch : {}, tr_loss: {:.3f}, val_loss: '
                       '{:.3f}, tr_acc: {:.2%}, val_acc: {:.2%}'.format(epoch + 1, tr_summ['loss'],
                                                                        val_summ['loss'],
                                                                        tr_summ['acc'], val_summ['acc']))

            val_loss = val_summ['loss']
            is_best = val_loss < best_val_loss

            if is_best:
                state = {'epoch': epoch + 1,
                         'model_state_dict': model.state_dict(),
                         'opt_state_dict': opt.state_dict()}
                summary = {'tr': tr_summ, 'val': val_summ}

                # manager.update_summary(summary)
                # manager.save_summary('summary.json')
                # manager.save_checkpoint(state, 'best.tar')

                best_val_loss = val_loss



if __name__ == '__main__':
    main()
