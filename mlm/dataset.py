from megatron.data.indexed_dataset import MMapIndexedDataset


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
                np_array = self.multi_dataset[p][idx if p == 0 else idx - self.acc[p]]
                return self.tokenizer(self.tokenizer.decode(np_array),
                                      padding=True,
                                      truncation=True,
                                      max_length=512,
                                      return_special_tokens_mask=True, )
