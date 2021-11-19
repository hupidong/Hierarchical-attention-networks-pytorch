"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from sent_att_model import SentAttNet
from word_att_model import WordAttNet
from utils import get_max_lengths
from dataset import MyDataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from sklearn import metrics
import numpy as np


# torch.set_num_threads(6)


def cross_entropy_loss(logits, labels):
    return F.cross_entropy(logits, labels)


class HierAttNet(pl.LightningModule):
    def __init__(self, word_hidden_size, sent_hidden_size, batch_size, num_classes, pretrained_word2vec_path,
                 max_sent_length, max_word_length, opt):
        super(HierAttNet, self).__init__()
        self.save_hyperparameters()
        self.batch_size = batch_size
        self.word_hidden_size = word_hidden_size
        self.sent_hidden_size = sent_hidden_size
        self.max_sent_length = max_sent_length
        self.max_word_length = max_word_length
        self.word_att_net = WordAttNet(pretrained_word2vec_path, word_hidden_size)
        self.sent_att_net = SentAttNet(sent_hidden_size, word_hidden_size, num_classes)
        self._init_hidden_state()
        self.opt = opt

    def _init_hidden_state(self, last_batch_size=None):
        if last_batch_size:
            batch_size = last_batch_size
        else:
            batch_size = self.batch_size
        self.word_hidden_state = torch.zeros(2, batch_size, self.word_hidden_size)
        self.sent_hidden_state = torch.zeros(2, batch_size, self.sent_hidden_size)

    def forward(self, input):
        output_list = []
        input = input.permute(1, 0, 2)
        for i in input:
            output, self.word_hidden_state = self.word_att_net(i.permute(1, 0), self.word_hidden_state)
            output_list.append(output)
        output = torch.cat(output_list, 0)
        output, self.sent_hidden_state = self.sent_att_net(output, self.sent_hidden_state)
        return output

    def training_step(self, train_batch, batch_idx):
        feature, label = train_batch
        self._init_hidden_state()
        predictions = self.forward(feature)
        loss = cross_entropy_loss(predictions, label)
        self.log('train_loss', loss, prog_bar=True)
        log = {'train_loss': loss.detach()}
        return {'loss': loss, 'log': log}

    def validation_step(self, val_batch, batch_idx):
        feature, label = val_batch
        self._init_hidden_state(last_batch_size=len(label))
        predictions = self.forward(feature)
        loss = cross_entropy_loss(predictions, label)
        y_pred = np.argmax(predictions.detach().numpy(), -1)
        accuracy = metrics.accuracy_score(y_true=label.numpy(), y_pred=y_pred)
        self.log('val_loss', loss)
        self.log('val_accuracy', accuracy)
        log = {'val_loss': loss.detach(), 'val_accuracy': accuracy}
        return {'val_loss': loss, 'log': log}

    def test_step(self, test_batch, batch_idx):
        feature, label = test_batch
        self._init_hidden_state(last_batch_size=len(label))
        predictions = self.forward(feature)
        loss = cross_entropy_loss(predictions, label)
        y_pred = np.argmax(predictions.detach().numpy(), -1)
        accuracy = metrics.accuracy_score(y_true=label.numpy(), y_pred=y_pred)
        self.log('test_loss', loss)
        self.log('test_accuracy', accuracy)
        log = {'test_loss': loss.detach()}
        return {'loss': loss, 'log': log}

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.parameters()), lr=self.opt.lr,
                                    momentum=self.opt.momentum)
        return optimizer


class CustomMonitor(Callback):
    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        metrics = trainer.callback_metrics
        epoch = trainer.current_epoch
        print(f"Epoch {epoch} Done: {metrics}")


if __name__ == "__main__":
    # data
    from parameters import MY_PARAMETERS

    max_word_length, max_sent_length = get_max_lengths(MY_PARAMETERS.train_set)
    train_set = MyDataset(MY_PARAMETERS.train_set, MY_PARAMETERS.word2vec_path, max_sent_length, max_word_length)
    train_loader = DataLoader(train_set, batch_size=MY_PARAMETERS.batch_size, shuffle=True, drop_last=True)
    test_set = MyDataset(MY_PARAMETERS.test_set, MY_PARAMETERS.word2vec_path, max_sent_length, max_word_length)
    test_loader = DataLoader(test_set, batch_size=MY_PARAMETERS.batch_size, shuffle=False, drop_last=False)

    # train
    model = HierAttNet(MY_PARAMETERS.word_hidden_size, MY_PARAMETERS.sent_hidden_size, MY_PARAMETERS.batch_size,
                       train_set.num_classes, MY_PARAMETERS.word2vec_path, max_sent_length, max_word_length,
                       MY_PARAMETERS)
    trainer = pl.Trainer(callbacks=[CustomMonitor()], log_every_n_steps=2, num_sanity_val_steps=1, val_check_interval=0.1,
                         num_processes=8, max_epochs=5)
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=test_loader)
