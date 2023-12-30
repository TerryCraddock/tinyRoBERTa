# You may need to use the hugging face cli login to get this repo
import os
from datasets import load_dataset_builder
from datasets import get_dataset_split_names
from datasets import load_dataset
from tqdm.auto import tqdm

dataset_name = "nthngdy/oscar-mini"
x = dataset_name.split("/")
smallName = x[1]

ds_builder = load_dataset_builder('nthngdy/oscar-mini', 'unshuffled_deduplicated_en')

# Inspect dataset description
print(f"Dataset Description: {ds_builder.info.description}")

# Inspect dataset features
# print(f"Dataset Features: {ds_builder.info.features}")

# Get split names
print(f"Dataset Split Names: {get_dataset_split_names('nthngdy/oscar-mini', 'unshuffled_deduplicated_en')}")

# Load dataset
print(f"Loading {smallName} dataset")
dataset = load_dataset('nthngdy/oscar-mini', 'unshuffled_deduplicated_en')

# Remove unneeded columns from dataset
dataset = dataset.remove_columns(['id'])

print(dataset)
print(dataset['train'][0])

# Save dataset
text_data = []
file_count = 0

for sample in tqdm(dataset['train']):
    sample = sample['text'].replace('\n', '')
    text_data.append(sample)
    if len(text_data) == 10_000:
        with open(f'{os.getcwd()}/data/oscar-mini/en_{file_count}.txt', 'w', encoding='utf-8') as fp:
            fp.write('\n'.join(text_data))
        text_data = []
        file_count += 1

# in case num_rows % 10_000 != 0 ie: 420000 % 10_000 = 0 so we comment out 
# with open(f'en_{file_count.txt}', 'w', encoding='utf-8') as fp:
#     fp.write('\n'.join(text_data))


