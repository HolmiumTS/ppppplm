# paths
TOKENIZER_PATH = '/run/media/holmium/DATA/TwitterPLM/tokenizer/'
RAW_DATA_PATH = '/run/media/holmium/DATA/TwitterPLM/data/'
MODEL_PATH = '/run/media/holmium/DATA/TwitterPLM/model/'
CHECKPOINT_PATH = '/run/media/holmium/DATA/TwitterPLM/checkpoint/'

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
VOCAB_SIZE = 5_000
MIN_FREQUENCY = 2
HIDDEN_SIZE = 20
NUM_HIDDEN_LAYERS = 2
NUM_ATTENTION_HEADS = 3
MAX_LEN = 64
