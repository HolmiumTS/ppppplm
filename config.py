# paths
TOKENIZER_PATH = './tokenizer/'
RAW_DATA_PATH = './data/'
MODEL_PATH = './model/'
CHECKPOINT_PATH = './checkpoint/'

# tokens
BOS_TOKEN = '<s>'
EOS_TOKEN = '</s>'
SEP_TOKEN = '</s>'
CLS_TOKEN = '<s>'
UNK_TOKEN = '<unk>'
PAD_TOKEN = '<pad>'
MASK_TOKEN = '<mask>'
SPECIAL_TOKENS = [
    '<s>',
    '</s>',
    '<unk>',
    '<pad>',
    '<mask>'
]

# hyper params
VOCAB_SIZE = 30522
MIN_FREQUENCY = 2
HIDDEN_SIZE = 768
NUM_HIDDEN_LAYERS = 12
NUM_ATTENTION_HEADS = 12
MAX_LEN =512
