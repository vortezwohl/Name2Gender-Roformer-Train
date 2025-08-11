# %% model initialize
import torch
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter

from data_preprocess import preprocess, train_dataset, test_dataset
from model import Name2GenderBase

model = Name2GenderBase()

# %% data process
train_token_ids, train_labels = preprocess(train_dataset, batched=True)
test_token_ids, test_labels = preprocess(test_dataset, batched=True)

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
        for step, (token_ids, label) in enumerate(zip(train_token_ids, train_labels)):
            train_step += 1
            output = model(token_ids)
            loss = loss_function(output, label.unsqueeze(dim=0))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(f'- (Train) Epoch [{epoch}/{num_epochs}], Step [{step}/{len(train_token_ids)}], Loss: {loss.item()}')
            writer.add_scalar('train/loss', loss.item(), train_step)
        for step, (token_ids, label) in enumerate(zip(test_token_ids, test_labels)):
            valid_step += 1
            with torch.no_grad():
                output = model(token_ids)
                loss = loss_function(output, label)
                print(f'- (Valid) Step [{step}/{len(test_token_ids)}], Loss: {loss.item()}')
                writer.add_scalar('valid/loss', loss.item(), valid_step)
except KeyboardInterrupt:
    model.save(model_dir='checkpoint')
