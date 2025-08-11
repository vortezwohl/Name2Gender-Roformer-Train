import os
import pickle
import time

import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from datasets import load_dataset

from model import Name2GenderBase, tokenizer

model = Name2GenderBase()
model.train()


def encode(text: str, batched: bool = False) -> torch.Tensor:
    if batched:
        return torch.tensor([tokenizer.encode(text)], dtype=torch.long, device=model.device)
    return torch.tensor(tokenizer.encode(text), dtype=torch.long, device=model.device)


ds = load_dataset("erickrribeiro/gender-by-name")
train_dataset, test_dataset = ds['train'], ds['test']

# 0 for female 1 for male
train_names = train_dataset.to_dict()['Name']
train_genders = train_dataset.to_dict()['Gender']
test_names = test_dataset.to_dict()['Name']
test_genders = test_dataset.to_dict()['Gender']

male_names, female_names = [], []
for i, name in enumerate(train_names):
    if train_genders[i] == 0:
        female_names.append(name)
    else:
        male_names.append(name)


os.makedirs('./data', exist_ok=True)
if os.path.exists('./data/train_loader.pkl') and os.path.exists('./data/valid_loader.pkl'):
    with open('./data/train_loader.pkl', 'rb') as f:
        train_dataset_loader = pickle.load(f)
    with open('./data/valid_loader.pkl', 'rb') as f:
        valid_dataset_loader = pickle.load(f)
else:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_ratio = 0.9
    batch_size = 16
    all_texts = male_names + female_names
    token_ids = [encode(x) for x in all_texts]
    labels = ([torch.tensor([1.], dtype=model.dtype, device=device) for _ in range(len(male_names))]
              + [torch.tensor([.0], dtype=model.dtype, device=device) for _ in range(len(female_names))])
    inputs = torch.stack(token_ids).to(device)
    labels = torch.stack(labels).to(device)
    dataset_size = len(labels)
    train_size = int(train_ratio * dataset_size)
    train_dataset = TensorDataset(inputs[:train_size], labels[:train_size])
    valid_dataset = TensorDataset(inputs[train_size:], labels[train_size:])
    train_dataset_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_dataset_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)
    with open('./data/train_loader.pkl', 'wb') as f:
        pickle.dump(train_dataset_loader, f)
    with open('./data/valid_loader.pkl', 'wb') as f:
        pickle.dump(valid_dataset_loader, f)


train_step = 0
valid_step = 0
acc_train_loss = 0.
acc_valid_loss = 0.
eval_interval = 10000
log_interval = 200


writer = SummaryWriter()
alpha = 1e-4
rho = 0.2
num_epochs = 1500
loss_function = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=2e-7)

try:
    for epoch in range(num_epochs):
        model.train()
        for batch_texts, batch_labels in train_dataset_loader:
            outputs = model.forward(batch_texts)
            loss = loss_function(outputs, batch_labels) + model.elastic_net(alpha=alpha, rho=rho)
            acc_train_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if train_step % log_interval == 0 and train_step > 0:
                writer.add_scalar('train/loss', acc_train_loss / log_interval, train_step)
                print(f'- Epoch {epoch} - Train Step {train_step} Loss {acc_train_loss / log_interval}', flush=True)
                acc_train_loss = 0.
            if train_step % eval_interval == 0:
                model.eval()
                with torch.no_grad():
                    for batch_texts, batch_labels in valid_dataset_loader:
                        outputs = model.forward(batch_texts)
                        loss = loss_function(outputs, batch_labels)
                        acc_valid_loss += loss.item()
                        if valid_step % len(valid_dataset_loader) == 0 and valid_step > 0:
                            writer.add_scalar('valid/loss', acc_valid_loss / len(valid_dataset_loader), valid_step)
                            print(f'- Epoch {epoch} - Valid Step {valid_step} Loss {acc_valid_loss / len(valid_dataset_loader)}', flush=True)
                            acc_valid_loss = 0.
                            # eval and save
                            acc_count = .0
                            model.eval()
                            with torch.no_grad():
                                for i, name in enumerate(test_names):
                                    if round(model(encode(name, batched=True)).item()) == test_genders[i]:
                                        acc_count += 1.
                            acc = 100 * (acc_count / len(test_names))
                            print(f'Accuracy: {acc:.4f}%', flush=True)
                            model.save(f'{int(time.time())}-ACC={acc:.2f}-{model._model_name}', model_dir='checkpoint')
                        valid_step += 1
                model.train()
            train_step += 1
except KeyboardInterrupt:
    acc_count = .0
    model.eval()
    with torch.no_grad():
        for i, name in enumerate(test_names):
            if round(model(encode(name, batched=True)).item()) == test_genders[i]:
                acc_count += 1.
    acc = 100 * (acc_count / len(test_names))
    print(f'Accuracy: {acc:.4f}%', flush=True)
    model.save(f'{int(time.time())}-ACC={acc:.2f}-{model._model_name}', model_dir='checkpoint')
