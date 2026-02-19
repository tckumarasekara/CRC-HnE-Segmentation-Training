import abc
from argparse import ArgumentParser
import pytorch_lightning as pl
import torch
import numpy as np

from losses.FocalLosses import FocalLoss, Cyclical_FocalLoss


class UnetSuper(pl.LightningModule):
    """UnetSuper is a basic implementation of the LightningModule without any ANN modules
    It is a parent class which should not be used directly
    """
    def __init__(self, hparams, **kwargs):
        super(UnetSuper, self).__init__()

        self.num_classes = kwargs["num_classes"]
        self.metric = iou_fnc
        self.save_hyperparameters(hparams)
        self.args = kwargs

        if kwargs["flat_weights"]:
            self.weights = [1, 1, 1, 1, 1, 1, 1]
        else:
            self.weights = [0.001, 1, 1, 1, 1, 1, 1]

        if kwargs["loss"] == "FocalLoss":
            self.criterion = FocalLoss(apply_nonlin=None, alpha=self.weights, gamma=2.0)
        else:
            self.criterion = Cyclical_FocalLoss()

        self.criterion.cuda()
        self._to_console = False
        self._val_outputs = []


    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--num_workers', type=int, default=4, metavar='N', help='number of workers (default: 16)')
        parser.add_argument('--lr', type=float, default=0.003, help='learning rate (default: 0.003)')
        parser.add_argument('--gamma-factor', type=float, default=2.0, help='gamma factor (default: 2.0)')
        parser.add_argument('--weight-decay', type=float, default=1e-5, help='weight decay (default: 0.0002)')
        parser.add_argument('--epsilon', type=float, default=1e-16, help='epsilon (default: 1e-16)')
        parser.add_argument('--models', type=str, default="Unet", help='the wanted model')
        parser.add_argument('--training-batch-size', type=int, default=1, help='Input batch size for training')
        parser.add_argument('--test-batch-size', type=int, default=1, help='Input batch size for testing')
        parser.add_argument('--dropout-val', type=float, default=0, help='dropout_value for layers')
        parser.add_argument('--flat-weights', type=bool, default=False, help='set all weights to 0.01')
        parser.add_argument('--loss', type=str, default="FocalLoss")
        return parser


    @abc.abstractmethod
    def forward(self, x):
        """
        Implemented in the child class, defines the forward pass of the model
        """
        pass


    def loss(self, logits, labels):
        """
        Initializes the loss function

        :return: output - Initialized cross entropy loss function
        """
        labels = labels.long()
        return self.criterion(logits, labels)


    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        prob_mask = self.forward(x)

        loss = self.criterion(prob_mask, y.type(torch.long), self.current_epoch)

        # log loss (Lightning will average per epoch)
        self.log("train_avg_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        # log IoU (per batch → averaged automatically)
        iter_iou, iter_count = iou_fnc(torch.argmax(prob_mask, dim=1).float(), y, self.args['num_classes'])

        for i in range(self.args['num_classes']):
            self.log(f"train_iou_{i}",
                torch.tensor(iter_iou[i], device=self.device),
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )

        return loss



    def validation_step(self, test_batch, batch_idx):
        """
        Predicts on the test dataset to compute the current performance of the models.
        :param test_batch: Batch data
        :param batch_idx: Batch indices
        :return: output - Validation performance
        """

        output = {}
        x, y = test_batch
        prob_mask = self.forward(x)

        loss = self.criterion(prob_mask, y.type(torch.long), self.current_epoch)

        iter_iou, iter_count = iou_fnc(torch.argmax(prob_mask, dim=1).float(), y, self.args['num_classes'])

        for i in range(self.args['num_classes']):
            output['val_iou_' + str(i)] = torch.tensor(iter_iou[i])
            output['val_iou_cnt_' + str(i)] = torch.tensor(iter_count[i])

        output['val_loss'] = loss
        self._val_outputs.append(output)

        return output


    def on_validation_epoch_end(self):
        outputs = self._val_outputs
        val_avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean().item()

        val_iou_sum = torch.zeros(self.args['num_classes'])
        val_iou_cnt_sum = torch.zeros(self.args['num_classes'])

        for i in range(self.args['num_classes']):
            val_iou_sum[i] = torch.stack([x['val_iou_' + str(i)] for x in outputs]).sum()
            val_iou_cnt_sum[i] = torch.stack([x['val_iou_cnt_' + str(i)] for x in outputs]).sum()

        iou_scores = val_iou_sum / (val_iou_cnt_sum + 1e-10)
        iou_mean = iou_scores[~torch.isnan(iou_scores)].mean().item()

        self.log('val_avg_loss', val_avg_loss, sync_dist=True, on_step=False, on_epoch=True)
        self.log('val_mean_iou', iou_mean, sync_dist=True, on_step=False, on_epoch=True)

        for c in range(self.args['num_classes']):
            if val_iou_cnt_sum[c] == 0.0:
                iou_scores[c] = 0
            self.log(f'val_iou_{c}', iou_scores[c].item(), sync_dist=True, on_step=False, on_epoch=True)

        if self._to_console:
            print(f'Validation Epoch {self.current_epoch} ------------------------')
            print(f'Loss: {val_avg_loss:.6f}, Mean IoU: {iou_mean:.6f}')
            for c in range(self.args['num_classes']):
                print(f'class {c} IoU: {iou_scores[c].item():.6f}')

        self._val_outputs.clear()


    def test_step(self, test_batch, batch_idx):
        x, y = test_batch
        prob_mask = self.forward(x)

        loss = self.criterion(prob_mask, y.type(torch.long), self.current_epoch)

        # log test loss
        self.log("test_avg_loss", loss, on_step=False, on_epoch=True, sync_dist=True)

        iter_iou, iter_count = iou_fnc(torch.argmax(prob_mask, dim=1).float(), y, self.args['num_classes'])

        for i in range(self.args['num_classes']):
            self.log(
                f"test_iou_{i}",
                torch.tensor(iter_iou[i], device=self.device),
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )

        return loss


    def prepare_data(self):
        """
        Prepares the data for training and prediction
        """
        return {}


    def configure_optimizers(self):
        """
        Initializes the optimizer and learning rate scheduler

        :return: output - Initialized optimizer and scheduler
        """

        self.optimizer = torch.optim.AdamW(self.parameters(), lr=self.args['lr'])
        self.scheduler = {'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.1, patience=10, min_lr=1e-6, ),
            'monitor': 'train_avg_loss', }

        return [self.optimizer], [self.scheduler]


def iou_fnc(pred, target, n_classes=7):
    ious = []
    pred = pred.view(-1)
    target = target.view(-1)

    count = np.zeros(n_classes)

    for cls in range(0, n_classes):
        pred_inds = pred == cls
        target_inds = target == cls

        intersection = (pred_inds[target_inds]).long().sum().cpu().item()
        union = pred_inds.long().sum().cpu().item() + target_inds.long().sum().cpu().item() - intersection  # .data.cpu()[0] - intersection

        if union == 0:
            ious.append(0.0)
        else:
            count[cls] += 1
            ious.append(float(intersection) / float(max(union, 1)))

    return np.array(ious), count
