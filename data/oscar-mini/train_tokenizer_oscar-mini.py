import os
from pathlib import Path
from tokenizers import ByteLevelBPETokenizer

# get path for each file
print('Gathering files...')
paths = [str(x) for x in Path('./data/oscar-mini/').glob('*.txt')]

# test print
#print(f'First five paths {paths[:5]}')

# initilize tokenizer
tokenizer = ByteLevelBPETokenizer()

# train tokenizer
tokenizer.train(files=paths, vocab_size=30_522, min_frequency=2, # to shorten amount of data we use files=paths[:100] will use first 100 only
                special_tokens=[
                    '<s>', '<pad>', '</s>', '<unk>', '<mask>'
                ]) # 30_522 is standard bert size/dat these are roberta special tokens

# create save directory
isExist = os.path.exists('./data/oscar-mini/TinyBert')
if not isExist:
    os.mkdir('./data/oscar-mini/TinyBert')

# save tokenizer
tokenizer.save_model('./data/oscar-mini/TinyBert')

# to load our roberta tokenizer
# from transformers import RobertaTokenizerFast
# tokenizer = RobertaTokenizerFast.from_pretrained('./data/oscar-mini/TinyBert')
# tokenizer('Hello how are you?)
