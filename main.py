import argparse
import time
import math
import numpy as np
import torch
import torch.nn as nn

import data
from model import RNNModel

from visualize.dump import dump, dump_hiddens, dump_words
from utils import batchify, batchify_padded, get_batch, repackage_hidden

parser = argparse.ArgumentParser(description='PyTorch PennTreeBank RNN/LSTM Language Model')
parser.add_argument('--data', type=str, default='data/penn/',
                    help='location of the data corpus')
parser.add_argument('--emsize', type=int, default=400,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=1150,
                    help='number of hidden units per layer')
parser.add_argument('--lr', type=float, default=30,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=8000,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=80, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=70,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--dropouth', type=float, default=0.,
                    help='dropout for rnn layers (0 = no dropout)')
parser.add_argument('--dropouti', type=float, default=0.,
                    help='dropout for input embedding layers (0 = no dropout)')
parser.add_argument('--dropoute', type=float, default=0.,
                    help='dropout to remove words from embedding layer (0 = no dropout)')
parser.add_argument('--wdrop', type=float, default=0.)
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--nonmono', type=int, default=5,
                    help='random seed')
parser.add_argument('--cuda', action='store_false',
                    help='use CUDA')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='report interval')
randomhash = ''.join(str(time.time()).split('.'))
parser.add_argument('--save', type=str,  default=randomhash+'.pt',
                    help='path to save the final model')
parser.add_argument('--alpha', type=float, default=0,
                    help='alpha L2 regularization on RNN activation (alpha = 0 means no regularization)')
parser.add_argument('--beta', type=float, default=0,
                    help='beta slowness regularization applied on RNN activiation (beta = 0 means no regularization)')
parser.add_argument('--wdecay', type=float, default=1.2e-6,
                    help='weight decay applied to all weights')
parser.add_argument('--resume', type=str,  default='',
                    help='path of model to resume')
parser.add_argument('--optimizer', type=str,  default='sgd',
                    help='optimizer to use (sgd, adam)')
parser.add_argument('--when', nargs="+", type=int, default=[-1],
                    help='When (which epochs) to divide the learning rate by 10 - accepts multiple')

# new arguments
parser.add_argument('--temperature', type=float, default=-1,    # < 0 is exp(-d()^2) and > 0 is exp(d()^2)
                    help='temperature in the exponent of the softmax.')
parser.add_argument('--nsamples', type=int, default=10,
                    help='number of negative samples.')

parser.add_argument('--dist_fn', type=str, default='eucl')
parser.add_argument('--activation_fn', type=str, default='logsoftmax')
parser.add_argument('--no_bias', action='store_false')
parser.add_argument('--uni_freq', action='store_true')
parser.add_argument('--reinit_h', action='store_true')
parser.add_argument('--bias_reg', type=float, default=0.)
parser.add_argument('--evaluate_every', type=int, default=1)

# dump settings
parser.add_argument('--dump_hiddens', action='store_true')
parser.add_argument('--dump_words', action='store_true')
parser.add_argument('--dump_valloss', type=str, default='valloss')
parser.add_argument('--dump_entropy', type=str, default='entropy_')

args = parser.parse_args()
args.tied = True

def run(args):

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        if not args.cuda:
            print("WARNING: You have a CUDA device, so you should probably run with --cuda")
        else:
            torch.cuda.manual_seed(args.seed)

    ###############################################################################
    # Load data
    ###############################################################################

    def model_save(fn):
        with open(fn, 'wb') as f:
            torch.save([model, optimizer], f)

    def model_load(fn):
        global model, criterion, optimizer
        with open(fn, 'rb') as f:
            model, optimizer = torch.load(f)

    import os
    import hashlib
    fn = 'corpus.{}.data'.format(hashlib.md5(args.data.encode()).hexdigest())
    if os.path.exists(fn):
        print('Loading cached dataset...')
        corpus = torch.load(fn)
    else:
        print('Producing dataset...')
        corpus = data.Corpus(args.data)
        torch.save(corpus, fn)

    # get token frequencies and eos_tokens
    frequencies, eos_tokens = None, None
    if not args.uni_freq: frequencies = corpus.frequencies
    if args.reinit_h: eos_tokens = corpus.reset_idxs

    # batchify
    eval_batch_size = 1
    test_batch_size = 1
    print(corpus.dictionary)
    if args.reinit_h:
        ntokens = len(corpus.dictionary) + 1 if args.batch_size > 1 else len(corpus.dictionary)
        train_data, seq_lens = batchify_padded(corpus.train, args.batch_size, args, ntokens, eos_tokens)    
    else:
        ntokens = len(corpus.dictionary)
        train_data = batchify(corpus.train, args.batch_size, args)
    val_data = batchify(corpus.valid, eval_batch_size, args)
    test_data = batchify(corpus.test, test_batch_size, args)

    ###############################################################################
    # Build the model
    ###############################################################################

    model = RNNModel(ntokens, args.emsize, args.nhid, args.dropout, args.dropouth, args.dropouti, args.dropoute, args.wdrop, args.nsamples,
                    args.temperature, frequencies, args.no_bias, args.bias_reg, args.dist_fn, args.activation_fn)
    ###
    if args.resume:
        print('Resuming model ...')
        model_load(args.resume)
        optimizer.param_groups[0]['lr'] = args.lr
        model.dropouti, model.dropouth, model.dropout, args.dropoute = args.dropouti, args.dropouth, args.dropout, args.dropoute

    ###
    if args.cuda:
        model = model.cuda()

    ###
    params = list(model.parameters())
    total_params = sum(x.size()[0] * x.size()[1] if len(x.size()) > 1 else x.size()[0] for x in params if x.size())
    print('Args:', args)
    print('Model total parameters:', total_params)

    ###############################################################################
    # Training code
    ###############################################################################

    def evaluate(data_source, epoch, batch_size=1):
        # Turn on evaluation mode which disables dropout.
        model.eval()

        if args.dump_hiddens:
            loss, entropy, hiddens = model.evaluate(data_source, eos_tokens, args.dump_hiddens)
            dump_hiddens(hiddens, 'hiddens_' + str(epoch))
        else:
            loss, entropy = model.evaluate(data_source, eos_tokens)
        
        if args.dump_words:
            dump_words(model.encoder.weight.detach().cpu().numpy(), 'words_' + str(epoch))

        if not args.dump_entropy is None:
            dump(entropy, args.dump_entropy + str(epoch))

        return loss


    def train():
        # Turn on training mode which enables dropout.
        total_loss, avrg_loss = 0, 0
        start_time = time.time()
        ntokens = len(corpus.dictionary)
        batch, i = 0, 0
        hidden = model.init_hidden(args.batch_size)
        while i < train_data.size(0)-1:

            if args.reinit_h:
                seq_len = seq_lens[batch] - 1
            else:
                bptt = args.bptt if np.random.random() < 0.95 else args.bptt / 2.
                # Prevent excessively small or negative sequence lengths
                seq_len = max(5, int(np.random.normal(bptt, 5)))
                # prevent negative sequence lengths
                # There's a very small chance that it could select a very long sequence length resulting in OOM
                # seq_len = min(seq_len, args.bptt + 10)

            lr2 = optimizer.param_groups[0]['lr']
            optimizer.param_groups[0]['lr'] = lr2 * seq_len / args.bptt
            model.train()
            data = get_batch(train_data, i, args, seq_len=seq_len)

            # Starting each batch, we detach the hidden state from how it was previously produced.
            # If we didn't, the model would try backpropagating all the way to start of the dataset.
            reset_hidden = args.reinit_h
            if reset_hidden:
                hidden = model.init_hidden(args.batch_size)

            hidden = repackage_hidden(hidden)
            optimizer.zero_grad()

            #raw_loss = model.train_crossentropy(data, eos_tokens)
            raw_loss, hidden = model(data, hidden)

            loss = raw_loss
            '''
            See what we can do here! We don't need the regularization as it is implicit!

            # Activiation Regularization
            if args.alpha: loss = loss + sum(args.alpha * dropped_rnn_h.pow(2).mean() for dropped_rnn_h in dropped_rnn_hs[-1:])
            # Temporal Activation Regularization (slowness)
            if args.beta: loss = loss + sum(args.beta * (rnn_h[1:] - rnn_h[:-1]).pow(2).mean() for rnn_h in rnn_hs[-1:])
            '''
            loss.backward()

            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            if args.clip: torch.nn.utils.clip_grad_norm_(params, args.clip)
            optimizer.step()

            total_loss += loss.data
            optimizer.param_groups[0]['lr'] = lr2
            if batch % args.log_interval == 0 and batch > 0:
                cur_loss = total_loss.item() / args.log_interval
                elapsed = time.time() - start_time
                print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:05.5f} | ms/batch {:5.2f} | '
                        'loss {:5.2f} | ppl {:8.2f} | bpc {:8.3f}'.format(
                    epoch, batch, len(train_data) // args.bptt, optimizer.param_groups[0]['lr'],
                    elapsed * 1000 / args.log_interval, cur_loss, cur_loss, cur_loss / math.log(2)))
                avrg_loss = avrg_loss + total_loss
                total_loss = 0
                start_time = time.time()
            ###
            batch += 1
            i += seq_len + 1

        return avrg_loss / train_data.size(0)

    # Loop over epochs.
    lr = args.lr
    best_val_loss = []
    valid_loss = []
    stored_loss = 100000000

    # At any point you can hit Ctrl + C to break out of training early.
    try:
        optimizer = None
        # Ensure the optimizer is optimizing params, which includes both the model's weights as well as the criterion's weight (i.e. Adaptive Softmax)
        if args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(params, lr=args.lr, weight_decay=args.wdecay)
        if args.optimizer == 'adam':
            optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.wdecay)
        for epoch in range(1, args.epochs+1):
            epoch_start_time = time.time()
            train_loss = train()
            _, s, _= np.linalg.svd(model.rnn.module.weight_hh_l0.cpu().detach().numpy())
            print(s[0])
            #dump(model.decoder.bias.cpu().detach().numpy(), 'bias_' + str(epoch) +'.out')
            
            # skip to beginning if not in evaluation mode
            if epoch % args.evaluate_every > 0:
                print('-' * 89)
                print('| end of epoch {:3d} | time: {:5.2f}s | train loss {:5.2f} |'.format(
                        epoch, (time.time() - epoch_start_time), train_loss))
                print('-' * 89) 
                continue

            # evaluate validation loss 
            if 't0' in optimizer.param_groups[0]:
                tmp = {}
                for prm in model.parameters():
                    #if 'ax' in optimizer.state[prm]:
                    tmp[prm] = prm.data.clone()
                    if 'ax' in optimizer.state[prm]:
                        prm.data = optimizer.state[prm]['ax'].clone()

                val_loss2 = evaluate(val_data, epoch)
                valid_loss.append(val_loss2)
                print('-' * 89)
                print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                    'valid ppl {:8.2f} | valid bpc {:8.3f}'.format(
                        epoch, (time.time() - epoch_start_time), val_loss2, math.exp(val_loss2), val_loss2 / math.log(2)))
                print('-' * 89)

                if val_loss2 < stored_loss:
                    model_save(args.save)
                    print('Saving Averaged!')
                    stored_loss = val_loss2

                for prm in model.parameters():
                    prm.data = tmp[prm].clone()

            else:
                val_loss = evaluate(val_data, epoch, eval_batch_size)
                valid_loss.append(val_loss)
                print('-' * 89)
                print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                    'valid ppl {:8.2f} | valid bpc {:8.3f}'.format(
                  epoch, (time.time() - epoch_start_time), val_loss, math.exp(val_loss), val_loss / math.log(2)))
                print('-' * 89)

                if val_loss < stored_loss:
                    model_save(args.save)
                    print('Saving model (new best validation)')
                    stored_loss = val_loss

                if args.optimizer == 'sgd' and 't0' not in optimizer.param_groups[0] and (len(best_val_loss)>args.nonmono and val_loss > min(best_val_loss[:-args.nonmono])):
                    print('Switching to ASGD')
                    optimizer = torch.optim.ASGD(model.parameters(), lr=args.lr, t0=0, lambd=0., weight_decay=args.wdecay)

                if epoch in args.when:
                    print('Saving model before learning rate decreased')
                    model_save('{}.e{}'.format(args.save, epoch))
                    print('Dividing learning rate by 10')
                    optimizer.param_groups[0]['lr'] /= 10.

                best_val_loss.append(val_loss)

    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')

    # Load the best saved model.
    model_load(args.save)

    # Run on test data.
    test_loss = evaluate(test_data, args.epochs+1, test_batch_size)
    print('=' * 89)
    print('| End of training | test loss {:5.2f} | test ppl {:8.2f} | test bpc {:8.3f}'.format(
        test_loss, math.exp(test_loss), test_loss / math.log(2)))
    print('=' * 89)

    return np.array(valid_loss), test_loss

'''
    ### MAIN ###
'''

#valid_loss, test_loss = run(args)

l = [[('adam', 1e-4), ('adam', 1e-3), ('sgd', 1) , ('sgd', 10)],
    [-100, -10, -1],
    ['eucl']]
args.dump_entropy = None
args.dump_valloss = None
import itertools
L = list(itertools.product(*l))
results = []
for (opt, lr), temp, dist_fn in L:

    settings = [opt, lr, temp, dist_fn]
    args.optimizer = opt
    args.lr = lr
    args.temperature = temp
    args.dist_fn = dist_fn

    valid_loss, test_loss = run(args)
    results.append(settings + [valid_loss])

for result in results:
    print(result)
