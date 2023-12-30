# Tiny RoBERTa Language Model

This project trains a small RoBERTa-based language model from scratch on the Oscar-mini dataset using the HuggingFace Transformers library and a custom tokenizer.

## Model Details

The model is a RoBERTa-style masked language model implemented using the HuggingFace `RobertaForMaskedLM` class. Key details include:

- Vocabulary size matched to Oscar-mini tokenizer (around 20k tokens)
- 6 hidden layers
- Hidden size of 768
- 12 attention heads
- Trained for 4 epochs by default

The model is trained using a masked language modeling (MLM) objective - during training, 15% of tokens are randomly masked and the model must predict the original tokens.

## Training

The model is trained using AdamW optimization and batch size of 1. The learning rate is set to 1e-4 by default.

The input data is tokenized using the custom Oscar-mini tokenizer and put into tensors. A masking function `mlm()` handles randomly masking input tokens for MLM.

The model and data are loaded onto a GPU accelerator if available, otherwise trained on CPU. A tqdm progress bar tracks loss over epochs.

## Usage

Key steps to train the model:

1. Install required packages:
   pip install -U -r requirements.txt
2. Get the oscar-mini dataset (You may need to use Hugging Face CLI login. If so you will be prompted)
   python data/oscar-mini/get_oscar-mini.py
3. Train the custom tokenizer
   python data/oscar-mini/train_tokenizer_oscar-mini.py
4. Train the model
   python train.py config/train_tiny.py

use a great configurator originaly created by Andrej Karpathy 
original code can be found here https://github.com/karpathy/nanoGPT
Thank you so much for this configurator I love it! This allows you to set hyperparameters and the such very easily. See config/train_tiny.py
