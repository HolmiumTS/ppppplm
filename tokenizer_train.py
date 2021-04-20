from tokenizers.implementations import ByteLevelBPETokenizer

import config
import os


def get_file():
    return [os.path.join(config.RAW_DATA_PATH, file) for file in os.listdir(config.RAW_DATA_PATH)]


def main():
    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(
        files=get_file(),
        vocab_size=config.VOCAB_SIZE,
        min_frequency=config.MIN_FREQUENCY,
        special_tokens=config.SPECIAL_TOKENS
    )

    tokenizer.save_model(config.TOKENIZER_PATH)


if __name__ == '__main__':
    main()
