from megatron.data.indexed_dataset import MMapIndexedDataset
from transformers import GPT2TokenizerFast

reverse_tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
reverse_tokenizer.add_special_tokens({
    'bos_token': '<s>',
    'eos_token': '</s>',
    'sep_token': '</s>',
    'cls_token': '<s>',
    'unk_token': '<unk>',
    'pad_token': '<pad>',
    'mask_token': '<mask>',
})

MAX_LEN = 512


class Dataset():
    def __init__(self, tokenizer, paths):
        if isinstance(paths, str):
            paths = [paths]
        self.tokenizer = tokenizer

        self.multi_dataset = [MMapIndexedDataset(path) for path in paths]
        self.acc = [len(ds) for ds in self.multi_dataset]
        self.num = len(self.acc)
        for i in range(1, self.num):
            self.acc[i] += self.acc[i - 1]

    def __len__(self):
        return self.acc[-1]

    def __getitem__(self, idx):
        for p in range(self.num):
            if self.acc[p] > idx:
                idx = idx if p == 0 else idx - self.acc[p]
                np_array = self.multi_dataset[p][idx][:-1]
                res = self.tokenizer(reverse_tokenizer.decode(np_array),
                                     padding=True,
                                     truncation=True,
                                     max_length=MAX_LEN,
                                     return_special_tokens_mask=True, )
                while len(res['input_ids']) < MAX_LEN:
                    if idx + 1 == self.acc[p]:
                        if p + 1 == self.num:
                            break
                        idx = 0
                        p += 1
                    else:
                        idx += 1
                    np_array = self.multi_dataset[p][idx][:-1]
                    tmp = self.tokenizer(reverse_tokenizer.decode(np_array),
                                         padding=True,
                                         truncation=True,
                                         max_length=MAX_LEN,
                                         return_special_tokens_mask=True, )
                    if len(res['input_ids']) + len(tmp['input_ids']) - 1 > MAX_LEN:
                        break

                    for x in res.keys():
                        res[x] += tmp[x][1:]
                return res
