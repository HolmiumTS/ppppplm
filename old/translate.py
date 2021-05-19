import json


def main():
    vocab_file = vocab_file
    merges_file = merges_file

    self.encoder = {}
    self.encoder[self.bos_token] = 0
    self.encoder[self.pad_token] = 1
    self.encoder[self.eos_token] = 2
    self.encoder[self.unk_token] = 3

    self.add_from_file(vocab_file)

    self.decoder = {v: k for k, v in self.encoder.items()}

    with open(merges_file, encoding="utf-8") as merges_handle:
        merges = merges_handle.read().split("\n")[:-1]
    merges = [tuple(merge.split()[:-1]) for merge in merges]
    self.bpe_ranks = dict(zip(merges, range(len(merges))))
    self.cache = {}


def add_from_file(self, f):
    """
    Loads a pre-existing dictionary from a text file and adds its symbols to this instance.
    """
    if isinstance(f, str):
        try:
            with open(f, "r", encoding="utf-8") as fd:
                self.add_from_file(fd)
        except FileNotFoundError as fnfe:
            raise fnfe
        except UnicodeError:
            raise Exception(f"Incorrect encoding detected in {f}, please rebuild the dataset")
        return

    lines = f.readlines()
    for lineTmp in lines:
        line = lineTmp.strip()
        idx = line.rfind(" ")
        if idx == -1:
            raise ValueError("Incorrect dictionary format, expected '<token> <cnt>'")
        word = line[:idx]
        self.encoder[word] = len(self.encoder)


if __name__ == '__main__':
    main()
