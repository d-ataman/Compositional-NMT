import torchtext
import torch.nn.functional as F
from torch.autograd import Variable
import torch
import six


def segment(inputbatch):
    newbatch = []
    lengths = []
    for line in inputbatch:
        sen = []
        for w in line:
            trigrams = []
            l = int(w.__len__())
            for i in range(0, l):
                if l == 1:
                    trigram = ''.join(('^', w[0], '$'))
                    trigrams.append(trigram)
                else:
                    if i == 0:
                        trigram = ''.join(('^', w[0], w[1]))
                        trigrams.append(trigram)
                    else:
                        if i == l - 1:
                            trigram = ''.join((w[l - 2], w[l - 1], '$'))
                            trigrams.append(trigram)
                        else:
                            trigram = ''.join((w[i - 1], w[i], w[i + 1]))
                            trigrams.append(trigram)
            sen.append(trigrams)
            lengths += [len(trigrams)]
        newbatch.append(sen)
 
    return tuple(newbatch), lengths


class WordBatch(object):
    """Defines a batch of examples along with its Fields.

    Attributes:
        batch_size: Number of examples in the batch.
        dataset: A reference to the dataset object the examples come from
            (which itself contains the dataset's Field objects).
        train: Whether the batch is from a training set.

    Also stores the Variable for each column in the batch as an attribute.
    """

    def __init__(self, data=None, dataset=None, device=None, train=True):
        """Create a Batch from a list of examples."""
        if data is not None:
            self.batch_size = len(data)
            self.dataset = dataset
            self.train = train
            for (name, field) in dataset.fields.items():
                if field is not None and name == 'src':
                    batch = [x.__dict__[name] for x in data]
                    if name == 'src':
                        newbatch, _ = segment(batch)
                        idx_batch = [field.process(x, device=device, train=train) for x in newbatch]
                        if isinstance(idx_batch[0], tuple):
                            lengths = [x[1] for x in idx_batch]
                            idx_batch = [x[0] for x in idx_batch]
                            max_char_len = max([x.size(0) for x in idx_batch])
                            seq_lens = [x.size(1) for x in idx_batch]
                            max_seq_len = max(seq_lens)
                            idx_batch = [F.pad(x, (0, max_seq_len - x.size(1), 0, max_char_len-x.size(0)), 'constant', 1) for x in idx_batch]
                            lengths = [F.pad(x, (0,  max_seq_len-x.size(0)), 'constant', max_char_len) for x in lengths]
                        #else:
                        #    max_char_len = max([x.size(0) for x in idx_batch])
                        #    max_seq_len = max([x.size(1) for x in idx_batch])
                        #    idx_batch = [F.pad(x, (0, max_seq_len - x.size(1), 0, max_char_len-x.size(0)), 'constant', 1) for x in idx_batch]
                            
                        idx_batch = torch.cat(idx_batch, dim=-1)
                        setattr(self, name, (idx_batch, (torch.cat(lengths), torch.cuda.LongTensor(seq_lens))))
                elif field is not None:
                    batch = [x.__dict__[name] for x in data]
                    setattr(self, name, field.process(batch, device=device, train=train))


    @classmethod
    def fromvars(cls, dataset, batch_size, train=True, **kwargs):
        """Create a Batch directly from a number of Variables."""
        batch = cls()
        batch.batch_size = batch_size
        batch.dataset = dataset
        batch.train = train
        for k, v in kwargs.items():
            setattr(batch, k, v)
        return batch


class WordIterator(torchtext.data.Iterator):

    def __init__(self, dataset, batch_size, sort_key=None, device=None,
                 batch_size_fn=None, train=True,
                 repeat=None, shuffle=None, sort=None,
                 sort_within_batch=None):
        super(WordIterator, self).__init__(dataset, batch_size, sort_key, device,
                                           batch_size_fn, train, repeat, shuffle,
                                           sort, sort_within_batch)

    def __iter__(self):
        while True:
            self.init_epoch()
            for idx, minibatch in enumerate(self.batches):
                # fast-forward if loaded from state
                if self._iterations_this_epoch > idx:
                    continue
                self.iterations += 1
                self._iterations_this_epoch += 1
                if self.sort_within_batch:
                    # NOTE: `rnn.pack_padded_sequence` requires that a minibatch
                    # be sorted by decreasing order, which requires reversing
                    # relative to typical sort keys
                    if self.sort:
                        minibatch.reverse()
                    else:
                        minibatch.sort(key=self.sort_key, reverse=True)
                yield WordBatch(minibatch, self.dataset, self.device,
                            self.train)
            if not self.repeat:
                raise StopIteration

