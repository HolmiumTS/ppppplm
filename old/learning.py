from transformers import RobertaTokenizerFast
import config
import os

tokenizer = RobertaTokenizerFast(
    os.path.join(config.TOKENIZER_PATH, 'vocab.json'),
    os.path.join(config.TOKENIZER_PATH, 'merges.txt'),
    bos_token=config.BOS_TOKEN,
    eos_token=config.EOS_TOKEN,
    sep_token=config.SEP_TOKEN,
    cls_token=config.CLS_TOKEN,
    unk_token=config.UNK_TOKEN,
    pad_token=config.PAD_TOKEN,
    mask_token=config.MASK_TOKEN
)
# tokenizer.enable_truncation(max_length=512)
print(
    tokenizer.convert_ids_to_tokens(tokenizer.encode('Hello world World WorLd'))
)
print(
    tokenizer('Hello world World WorLd')
)
