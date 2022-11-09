import copy
from turtle import pos

import torch
import pytorch_lightning as pl
from src.models.layer import *
from src.utils import clones
from pytorch_lightning.callbacks import Callback
import torchmetrics
import torch.nn.functional as F

class periodicTransformer(pl.LightningModule):
    def __init__(self, n_classes=5, d_model=200, d_ff=128, h=8, N=4, time='discrete'):
        super().__init__()
        self.time = time
        self.pos_enc_discrete = PositionalEncoding(d_model)
        self.pos_enc_continuous = TimeFilm(embedding_size=d_model)
        self.enc_blocks = clones(EncoderBlock(d_model=d_model, d_ff=d_ff, h=h), N)
        self.proj = nn.Linear(d_model, n_classes)
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform(p)
        # Metrics -----------
        self.val_acc = torchmetrics.Accuracy()
        self.test_f1 = torchmetrics.F1Score(number_classes=n_classes, average="micro")
        self.test_acc = torchmetrics.Accuracy()
        self.test_recall = torchmetrics.Recall(num_classes=n_classes)
        self.test_precision = torchmetrics.Precision(num_classes=n_classes)
        self.test_preds = []
        self.test_targ = []
        self = self.double()

    def forward(self, x, t):
        if self.time == 'continuous':
            x = self.pos_enc_continuous(x, t)
        else:
            x = self.pos_enc_discrete(x)
        for enc in self.enc_blocks:
            x = F.relu(enc(x))
        # x = self.proj(x)
        # F.log_softmax(x, dim=-1)
        return self.proj(x)

    def loss_function(self, data, label):
        criterion = torch.nn.CrossEntropyLoss()
        return criterion(data, label)

    def training_step(self, batch, batch_idx):
        x, t, y = batch['mag'], batch['mjd'], batch['label']
        output = self(x.unsqueeze(dim=2), t.unsqueeze(dim=2))
        loss = self.loss_function(output[:,-1], y)
        self.log('loss',loss)
        return {'loss':loss}

    def validation_step(self, batch, batch_idx):
        x, t, y = batch['mag'], batch['mjd'], batch['label']
        output = self(x.unsqueeze(dim=2), t.unsqueeze(dim=2))
        val_loss = self.loss_function(output[:,-1], y)
        val_preds = self.predictions(output, y)
        val_acc = self.val_acc(val_preds, y)
        self.log('val_loss', val_loss)
        self.log('val_acc', val_acc)
        return {'val_loss':val_loss, 'val_acc':val_acc}
    
    def on_save_checkpoint(self, checkpoint):
        self.best_epoch = self.current_epoch        

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4, betas=(0.9, 0.98), eps=1e-9)

    def test_step(self, batch, batch_idx):
        x, t, y = batch['mag'], batch['mjd'], batch['label']
        output = self(x.unsqueeze(dim=2), t.unsqueeze(dim=2))
        test_loss = self.loss_function(output[:,-1], y)
        test_preds = self.predictions(output, y)
        test_acc = self.test_acc(test_preds, y)
        test_f1 = self.test_f1(test_preds, y)
        test_recalll = self.test_recall(test_preds, y)
        test_precision = self.test_precision(test_preds, y)
        self.test_preds += test_preds.tolist()
        self.test_targ += y.tolist()
        self.log('test_loss', test_loss)
        self.log('test_acc', test_acc)
        self.log('test_recall', test_recalll)
        self.log('test_precision', test_precision)
        self.log('test_f1', test_f1)
        return {'test_loss':test_loss, 'test_acc':test_acc,
                'test_recall':test_recalll, 'test_precision': test_precision,
                'test_f1':test_f1}

    def predictions(self, x, y):
        prediction = F.log_softmax(x, dim=-1)
        prediction = prediction[:,-1].argmax(axis=1)
        return prediction


class MetricTracker(Callback):
    def __init__(self):
        self.loss_history = []
        self.val_loss_history = []
        self.val_acc_history = []
        self.test_loss_history = []
        self.test_acc_history = [] 
        self.test_f1_history = []
        self.test_recall_history = []
        self.test_precision_history = []
        
    def on_validation_epoch_end(self, trainer, module):
        elogs = trainer.logged_metrics['val_loss'].cpu()
        elogs_acc = trainer.logged_metrics['val_acc'].cpu()
        self.val_loss_history.append(elogs)
        self.val_acc_history.append(elogs_acc)
        
    def on_train_epoch_end(self, trainer, module):
        elogs = trainer.logged_metrics['loss'].cpu()
        self.loss_history.append(elogs)
    
    def on_test_epoch_end(self, trainer, module):
        elogs = trainer.logged_metrics['test_loss'].cpu()
        elogs_acc = trainer.logged_metrics['test_acc'].cpu()
        elogs_f1 = trainer.logged_metrics['test_f1'].cpu()
        elogs_recall = trainer.logged_metrics['test_recall'].cpu()
        elogs_precision = trainer.logged_metrics['test_precision'].cpu()
        self.test_loss_history.append(elogs)
        self.test_acc_history.append(elogs_acc)
        self.test_f1_history.append(elogs_f1)
        self.test_recall_history.append(elogs_recall)
        self.test_precision_history.append(elogs_precision)

# --------------- NO PL -------------------------- 
#
# class periodicTransformer(nn.Module):
#     def __init__(self, n_classes=5, d_model=200, d_ff=128, h=8):
#         super().__init__()
#         self.pos_enc = PositionalEncoding(d_model)
#         self.enc = EncoderBlock(d_model=d_model, d_ff=d_ff, h=h)
#         self.proj = nn.Linear(d_model, n_classes)
#         for p in self.parameters():
#             if p.dim() > 1:
#                 nn.init.xavier_uniform(p)

#     def forward(self, x):
#         x = self.pos_enc(x)
#         x = self.enc(x)
#         x = self.proj(x)
#         return F.log_softmax(x, dim=-1)

#     def loss_function(self, data, label):
#         criterion = torch.nn.CrossEntropyLoss()
#         return criterion(data, label)

#     def training_step(self, batch, batch_idx):
#         x, y = batch['mag'], batch['label']
#         output = self(x.unsqueeze(dim=2))
#         loss = self.loss_function(output.mean(axis=1).max(axis=1)[0], y)
#         return loss

#     def validation_step(self, batch, batch_idx):
#         x, y = batch['mag'], batch['label']
#         output = self(x.unsqueeze(dim=2))
#         loss = self.loss_function(output.mean(axis=1).max(axis=1)[0], y)
#         _, acc = self.evaluate(output, y)
#         return loss, acc

#     def configure_optimizers(self):
#         return torch.optim.Adam(self.parameters(), lr=1e-4, betas=(0.9, 0.98), eps=1e-9)

#     def fit(self, train_loader, val_loader, n_epochs):
#         optimizer = self.configure_optimizers()
#         loss_history = []
#         val_loss_history = []
#         save_best_model = SaveBestModel()
#         for epoch in range(n_epochs):
#             for idx, batch in enumerate(train_loader):
#                 optimizer.zero_grad()
#                 loss = self.training_step(batch, idx)
#                 loss.backward()
#                 optimizer.step()
#             for idx, batch in enumerate(val_loader):
#                 val_loss, val_acc = self.validation_step(batch, idx)
#             loss_history.append(loss.detach().item())
#             val_loss_history.append(val_loss.detach().item())
#             print(f'Epoch: {epoch} - Train loss: {loss} - Val loss: {val_loss} - Val acc: {val_acc}')
#             save_best_model(val_loss, epoch, self)
#         self.model = save_best_model.best_model
#         return loss_history, val_loss_history

#     def test(self, test_loader):
#         test_avg_acc = []
#         predictions = []
#         for idx, batch in enumerate(test_loader):
#             x, y = batch['mag'], batch['label']
#             output = self(x)
#             prediction, test_acc = self.evaluate(output, y)
#             test_avg_acc.append(test_acc)
#             predictions += prediction
#             print(f'Test batch acc: {test_acc}')
#         test_avg_acc = sum(test_avg_acc) / len(test_avg_acc)
#         print(f'Test avg acc: {test_avg_acc}')
#         return test_avg_acc, predictions

#     def evaluate(self, x, y):
#         prediction = x.mean(axis=1).argmax(axis=1)
#         accuracy = (prediction == y).sum() / y.shape[0]
#         return list(zip(y.tolist(),prediction.tolist())), accuracy




# -------------------- DEPRECATED -----------
#
# class periodicTransformer:
#     def __init__(self, n_classes, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1):
#         self.model = self.make_model(n_classes, N, d_model, d_ff, h, dropout)

#     def make_model(self, n_classes, N, d_model, d_ff, h, dropout):
#         "Helper: Construct a model from hyperparameters."
#         c = copy.deepcopy
#         attn = torch.nn.MultiheadAttention(d_model, h)
#         ff = PositionwiseFeedForward(d_model, d_ff, dropout)
#         position = PositionalEncoding(d_model, dropout)
#         encoder_layer = EncoderLayer(d_model, c(attn), c(ff), dropout)
#         model = Encoder(position, encoder_layer, N, d_model, n_classes)
        
#         # This was important from their code. 
#         # Initialize parameters with Glorot / fan_avg.
#         for p in model.parameters():
#             if p.dim() > 1:
#                 nn.init.xavier_uniform(p)
#         return model

#     def loss_function(self, data, label):
#         criterion = torch.nn.CrossEntropyLoss()
#         return criterion(data, label)

#     def training_step(self, batch, batch_idx):
#         x, y = batch['mag'], batch['label']
#         output = self.model(x)
#         print(output.shape)
#         loss = self.loss_function(output, y)
#         return loss

#     def validation_step(self, batch, batch_idx):
#         x, y = batch['mag'], batch['label']
#         output = self.model(x)
#         loss = self.loss_function(output, y)
#         _, acc = self.evaluate(output, y)
#         return loss, acc

#     def configure_optimizers(self):
#         return torch.optim.Adam(self.model.parameters(), lr=1e-4, betas=(0.9, 0.98), eps=1e-9)

#     def fit(self, train_loader, val_loader, n_epochs):
#         optimizer = self.configure_optimizers()
#         loss_history = []
#         val_loss_history = []
#         save_best_model = SaveBestModel()
#         for epoch in range(n_epochs):
#             for idx, batch in enumerate(train_loader):
#                 optimizer.zero_grad()
#                 loss = self.training_step(batch, idx)
#                 loss.backward()
#                 optimizer.step()
#             for idx, batch in enumerate(val_loader):
#                 val_loss, val_acc = self.validation_step(batch, idx)
#             loss_history.append(loss.detach().item())
#             val_loss_history.append(val_loss.detach().item())
#             print(f'Epoch: {epoch} - Train loss: {loss} - Val loss: {val_loss} - Val acc: {val_acc}')
#             save_best_model(val_loss, epoch, self.model)
#         self.model = save_best_model.best_model
#         return loss_history, val_loss_history

#     def test(self, test_loader):
#         test_avg_acc = []
#         predictions = []
#         for idx, batch in enumerate(test_loader):
#             x, y = batch['mag'], batch['label']
#             output = self.model(x)
#             prediction, test_acc = self.evaluate(output, y)
#             test_avg_acc.append(test_acc)
#             predictions += prediction
#             print(f'Test batch acc: {test_acc}')
#         test_avg_acc = sum(test_avg_acc) / len(test_avg_acc)
#         print(f'Test avg acc: {test_avg_acc}')
#         return test_avg_acc, predictions

#     def evaluate(self, x, y):
#         prediction = x.argmax(dim=1)
#         accuracy = (prediction == y).sum() / y.shape[0]
#         return list(zip(y.tolist(),prediction.tolist())), accuracy