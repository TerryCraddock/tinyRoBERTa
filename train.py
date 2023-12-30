import torch
from pathlib import Path
from transformers import RobertaTokenizerFast
from transformers import RobertaConfig
from transformers import RobertaForMaskedLM
from tqdm.auto import tqdm
from transformers import AdamW

# import our custom tokenizer
tokenizer = RobertaTokenizerFast.from_pretrained('./data/oscar-mini/TinyBert')

batch_size = 1
hidden_size = 768
attention_heads = 12
hidden_layers = 6 #Roberta default is 12
max_epochs = 4


# -----------------------------------------------------------------------------
config_keys = [
    k
    for k, v in globals().items()
    if not k.startswith("_") and isinstance(v, (int, float, bool, str))
]
exec(open("configurator.py").read())  # overrides from command line or config file
config = {k: globals()[k] for k in config_keys}  # will be useful for logging
# -----------------------------------------------------------------------------

# Masking function
def mlm(tensor):
    rand = torch.rand(tensor.shape) # array in shape of input tensor [0, 1]
    mask_arr = (rand < 0.15) * (tensor > 2) # special tokens are 0, 1, 2
    for i in range(tensor.shape[0]):
        selection = torch.flatten(mask_arr[i].nonzero()).tolist()
        tensor[i, selection] = 4 # <mask> is 4 in vocab
    return tensor

# get path for each file
paths = [str(x) for x in Path('./data/oscar-mini/').glob('*.txt')]

# test print
# print(f'First five paths {paths[:5]}')

# Tensors
print('Building Tensors...')
input_ids = []
mask = []
labels = []

for path in tqdm(paths):
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.read().split('\n')
    sample = tokenizer(lines, max_length=512, padding='max_length', truncation=True, return_tensors='pt')
    labels.append(sample.input_ids)
    mask.append(sample.attention_mask)
    input_ids.append(mlm(sample.input_ids.detach().clone()))

input_ids = torch.cat(input_ids)
mask = torch.cat(mask)
labels = torch.cat(labels)

encodings = {
    'input_ids': input_ids,
    'attention_mask': mask,
    'labels': labels
}

class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings
    def __len__(self):
        return self.encodings['input_ids'].shape[0]
    def __getitem__(self, i):
        return{key: tensor[i] for key, tensor in self.encodings.items()}

# init dataset
dataset = Dataset(encodings)

# init dataloader
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

config = RobertaConfig(
    vocab_size=tokenizer.vocab_size,
    max_position_embeddings=514,
    hidden_size=hidden_size,
    num_attention_heads=attention_heads,
    num_hidden_layers=hidden_layers,
    type_vocab_size=1
)

# init model
model = RobertaForMaskedLM(config)
# set torch device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# set optim
optim = AdamW(model.parameters(), lr=1e-4)
# move model to device
model.to(device)

# training loop
epochs = max_epochs
step = 0
loop = tqdm(dataloader, leave=True)
for epoch in range(epochs):
    for batch in loop:
        optim.zero_grad()
        input_ids = batch['input_ids'].to(device)
        mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask=mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optim.step()
        loop.set_description(f'Epoch: {epoch}')
        loop.set_postfix(loss=loss.item())