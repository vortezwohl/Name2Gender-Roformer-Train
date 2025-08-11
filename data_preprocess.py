import torch
from datasets import load_dataset

from model import tokenizer

ds = load_dataset("erickrribeiro/gender-by-name")
train_dataset, test_dataset = ds['train'], ds['test']


def encode(text: str, batched: bool = False, device: torch.device = None) -> torch.Tensor:
    if batched:
        return torch.tensor([tokenizer.encode(text)], dtype=torch.long, device=device)
    return torch.tensor(tokenizer.encode(text), dtype=torch.long, device=device)


def preprocess(dataset: dict, dtype: torch.dtype = torch.float32, device: torch.device = torch.device('cpu')) -> list:
    names, genders = dataset['Name'], dataset['Gender']
    male_names, female_names = [], []
    for i, name in enumerate(names):
        if genders[i] == 0:
            female_names.append(name)
        else:
            male_names.append(name)
    all_texts = male_names + female_names
    token_ids = [encode(x, device=device, batched=True) for x in all_texts]
    labels = ([torch.tensor([1.], dtype=dtype, device=device) for _ in range(len(male_names))]
              + [torch.tensor([.0], dtype=dtype, device=device) for _ in range(len(female_names))])
    return list(zip(token_ids, labels))


if __name__ == '__main__':
    print(preprocess(test_dataset.to_dict()))
