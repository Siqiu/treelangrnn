import torch


def repackage_hidden(h):
    """Wraps hidden states in new Tensors,
    to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


def batchify(data, bsz, args):

    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    if args.cuda:
        data = data.cuda()
    return data

def batchify_padded(data, bsz, args, ntokens, eos_tokens):

    i, batches, seq_lens = 0, [], []
    while i < data.size(0):

        # get the sentences
        j, sentences = i, [i]
        while j < data.size(0) and len(sentences) < bsz+1:
            if int(data[j].data.cpu().numpy()) in eos_tokens:
                sentences.append(j+1)
            j += 1

        i = j + 1

        # find longest sentence
        lengths = [sentences[j+1] - sentences[j] for j in range(len(sentences)-1)]
        longest = max(lengths)
        seq_lens.append(longest)

        # initialize empty container
        batch = torch.ones(longest, bsz) * ntokens
        for j in range(len(sentences)-1):
            print(length[j])
            print(sentences[j:j+1])
            batch[0:lengths[j]][j] = data[sentences[j]:sentences[j+1]]

        batches.append(batch)

    return torch.cat(batches, 0), seq_lens


def get_batch(source, i, args, seq_len=None, evaluation=False, eos_tokens=None):

    if eos_tokens is None:
        seq_len = min(seq_len if seq_len else args.bptt, len(source) - 1 - i)
        data = source[i:i+1+seq_len]

    return data
