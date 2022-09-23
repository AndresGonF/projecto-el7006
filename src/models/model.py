import copy

import torch

from src.models.layer import *
from src.utils import SaveBestModel

class periodicTransformer:
    def __init__(self, n_classes, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1):
        self.model = self.make_model(n_classes, N, d_model, d_ff, h, dropout)

    def make_model(self, n_classes, N, d_model, d_ff, h, dropout):
        "Helper: Construct a model from hyperparameters."
        c = copy.deepcopy
        attn = torch.nn.MultiheadAttention(d_model, h)
        ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        position = PositionalEncoding(d_model, dropout)
        encoder_layer = EncoderLayer(d_model, c(attn), c(ff), dropout)
        model = Encoder(encoder_layer, N, d_model, n_classes)
        
        # This was important from their code. 
        # Initialize parameters with Glorot / fan_avg.
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform(p)
        return model

    def loss_function(self, data, label):
        criterion = torch.nn.CrossEntropyLoss()
        return criterion(data, label)

    def training_step(self, batch, batch_idx):
        x, y = batch['mag'], batch['label']
        output = self.model(x)
        loss = self.loss_function(output, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch['mag'], batch['label']
        output = self.model(x)
        loss = self.loss_function(output, y)
        _, acc = self.evaluate(output, y)
        return loss, acc

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=1e-4, betas=(0.9, 0.98), eps=1e-9)

    def fit(self, train_loader, val_loader, n_epochs):
        optimizer = self.configure_optimizers()
        loss_history = []
        val_loss_history = []
        save_best_model = SaveBestModel()
        for epoch in range(n_epochs):
            for idx, batch in enumerate(train_loader):
                optimizer.zero_grad()
                loss = self.training_step(batch, idx)
                loss.backward()
                optimizer.step()
            for idx, batch in enumerate(val_loader):
                val_loss, val_acc = self.validation_step(batch, idx)
            loss_history.append(loss.detach().item())
            val_loss_history.append(val_loss.detach().item())
            print(f'Epoch: {epoch} - Train loss: {loss} - Val loss: {val_loss} - Val acc: {val_acc}')
            save_best_model(val_loss, epoch, self.model)
        self.model = save_best_model.best_model
        return loss_history, val_loss_history

    def test(self, test_loader):
        test_avg_acc = []
        predictions = []
        for idx, batch in enumerate(test_loader):
            x, y = batch['mag'], batch['label']
            output = self.model(x)
            prediction, test_acc = self.evaluate(output, y)
            test_avg_acc.append(test_acc)
            predictions += prediction
            print(f'Test batch acc: {test_acc}')
        test_avg_acc = sum(test_avg_acc) / len(test_avg_acc)
        print(f'Test avg acc: {test_avg_acc}')
        return test_avg_acc, predictions

    def evaluate(self, x, y):
        prediction = x.argmax(dim=1)
        accuracy = (prediction == y).sum() / y.shape[0]
        return list(zip(y.tolist(),prediction.tolist())), accuracy