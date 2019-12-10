import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from dataset import KeplerDataset
import pytorch_lightning as pl
from model import Encoder
from constants import config
from sklearn.metrics import *

class KeplerModel(pl.LightningModule):
    def __init__(self):
        super(KeplerModel, self).__init__()

        #Initialize Model Parameters Using Config Properties
        self.model = Encoder(config['seq_length'], config['hidden_size'], config['output_dim'], config['n_layers'])
        
        #Initialize a Cross Entropy Loss Criterion for Training
        self.criterion = torch.nn.CrossEntropyLoss()
    
    #Define a Forward Pass of the Model
    def forward(self, x, h):
        return self.model.forward(x, h)

    def training_step(self, batch, batch_idx):
        
        #Set Model to Training Mode
        self.model.train()
        
        #Unpack Data and Labels from Batch
        x, y = batch

        #Reshape Data into Shape (batch_size, 1, seq_length)
        x = x.view(x.size(0), -1, x.size(1))
        
        #Initalize the hidden state for forward pass
        h = self.model.init_hidden(x.size(0))

        #Zero out the model gradients to avoid accumulation
        self.model.zero_grad()

        #Forward Pass Through Model
        out, h = self.forward(x, h)

        #Calculate Cross Entropy Loss
        loss = self.criterion(out, y.long().squeeze())

        #Obtain Class Labels
        y_hat = torch.max(out, 1)[1]

        #Compute the balanced accuracy (weights based on number of ex. in each class)
        accuracy = balanced_accuracy_score(y, y_hat)

        #Compute weighted f1 score to account for class imbalance
        f1 = f1_score(y, y_hat, average='weighted')

        #Create metric object for tensorboard logging
        tensorboard_logs = {'train_loss': loss.item(), 'accuracy':accuracy, 'f1': f1}
        
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):

        #Set Model to Eval Mode
        self.model.eval()
        
        #Unpack data and labels from batch
        x, y = batch

        #Initialize Hidden State
        h = self.model.init_hidden(x.size(0))

        #Reshape Data into Shape (batch_size, 1, seq_length)
        x = x.view(x.size(0), -1, x.size(1))

        #Calculate Forward Pass of The Model
        out, h = self.forward(x, h)

        #Calculate Cross Entropy Loss
        loss = self.criterion(out, y.long().squeeze())

        #Calculate Class Indicies
        y_hat = torch.max(out, 1)[1]

        #Calculate Balanced Accuracy
        val_accuracy = torch.Tensor([balanced_accuracy_score(y, y_hat)])

        #Calculate Balanced Accuracy
        val_f1 = torch.Tensor([f1_score(y, y_hat, average='weighted')])

        #Create a metrics object
        metrics = {'val_loss': loss, 'val_accuracy':val_accuracy, 'val_f1': val_f1}

        return metrics

    def validation_end(self, outputs):
        # OPTIONAL
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['val_accuracy'] for x in outputs]).mean()
        avg_f1 = torch.stack([x['val_f1'] for x in outputs]).mean()

        tensorboard_logs = {'val_loss': avg_loss, 'val_acc':avg_acc, 'val_f1': avg_f1}
        
        return {'avg_val_loss': avg_loss, 'log': tensorboard_logs}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

    @pl.data_loader
    def train_dataloader(self):
        # REQUIRED
        return DataLoader(KeplerDataset(mode="train"), batch_size=64, shuffle=True)
    @pl.data_loader
    def val_dataloader(self):
        # REQUIRED
        return DataLoader(KeplerDataset(mode="test"), batch_size=128, shuffle=True)