import torch


def repackage_hidden(h):
    """Wraps hidden states in new Tensors,
    to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


def batchify(data, bsz, args, nsentences_of_length=None):

    # if the data is sorted by length we do a sorted batchify
    if not nsentences_of_length is None:
        return batchify_sorted(data, bsz, args, nsentences_of_length)

    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    if args.cuda:
        data = data.cuda()
    return data

def batchify_sorted(data, bsz, args, nsentences_of_length):
    '''
        this assumes that the dataset was sorted by length of the sentences
        such that it can batch sentences of same length together
    '''

    new_data, offset = None, 0
    for length, nsentences in nsentences_of_length:

        if nsentences < bsz:
            offset = offset + (nsentences*length)
            continue

        batchified = batchify(data[offset:offset + (nsentences*length)], bsz, args, None)
        
        if new_data is None:
            new_data = batchified
        else:
            new_data = torch.cat((new_data, batchified), 0)

    print(new_data)
    return new_data

def get_batch(source, i, args, seq_len=None, evaluation=False, eos_tokens=None):

    if eos_tokens is None:
        seq_len = min(seq_len if seq_len else args.bptt, len(source) - 1 - i)
        data = source[i:i+1+seq_len]

    return data
