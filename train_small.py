# %% model initialize
import time
import random

import torch
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter

from data_preprocess import preprocess, train_dataset, test_dataset
from model import Name2GenderSmall

model_name = f'name2gender-small.{int(time.time())}'
model = Name2GenderSmall(dtype=torch.float32)

# %% data process
train_set = preprocess(train_dataset, batched=True)
test_set = preprocess(test_dataset, batched=True)

# %% shuffle dataset
random.shuffle(train_set)
random.shuffle(test_set)

# %% train
train_step = 0
valid_step = 0
writer = SummaryWriter()
alpha = 1e-4
rho = 0.2
num_epochs = 1500
loss_function = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=2e-6)
try:
    for epoch in range(num_epochs):
        model.train()
        for step, (token_ids, label) in enumerate(train_set):
            train_step += 1
            output = model(token_ids)
            loss = loss_function(output, label.unsqueeze(dim=0))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(f'- (Train) Epoch [{epoch}/{num_epochs}], Step [{step}/{len(train_set)}], Loss: {loss.item()}')
            writer.add_scalar('train/loss', loss.item(), train_step)
        for step, (token_ids, label) in enumerate(test_set):
            valid_step += 1
            with torch.no_grad():
                output = model(token_ids)
                loss = loss_function(output, label.unsqueeze(dim=0))
                print(f'- (Valid) Step [{step}/{len(test_set)}], Loss: {loss.item()}')
                writer.add_scalar('valid/loss', loss.item(), valid_step)
except KeyboardInterrupt:
    model.save(model_name=model_name, model_dir='checkpoint')
model.save(model_name=model_name, model_dir='checkpoint')
