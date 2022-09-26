import sys
import os

import numpy as np
import torch
from tensorboardX import SummaryWriter
import mlflow
from sklearn.metrics import accuracy_score

from utils import train_utils
# TODO: improve metrics
# sys.path.append("..")
# from modules.utils import metrics


class Trainer(object):
    def __init__(self,
                 model, 
                 criterion, 
                 optimizer, 
                 train_dataloader, 
                 valid_dataloader,
                 logger,
                 device,
                 n_class,
                 exp_path,
                 train_epoch=10,
                 batch_size=1,
                 lr_scheduler=None,
                 valid_activation=None,
                 USE_TENSORBOARD=True,
                 USE_CUDA=True,
                 history=None,
                 checkpoint_saving_steps=20,
                 patience=10,
                 strict=False,
                 batch_transform=None,
                 batch_valid_transform=None,
                 ):
        self.n_class = n_class
        self.train_epoch = train_epoch
        self.batch_size = batch_size
        self.criterion = criterion
        self.optimizer = optimizer
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.logger = logger
        self.iterations = 0
        self.device = device
        self.history = history
        self.model = model
        self.strict = strict
        if history is not None:
            self.load_model_from_checkpoint(self.history, model_state_key='net')
        self.valid_activation = valid_activation
        self.USE_TENSORBOARD = USE_TENSORBOARD
        self.exp_path = exp_path
        if self.USE_TENSORBOARD:
            self.train_writer = SummaryWriter(log_dir=os.path.join(self.exp_path, 'train'))
            self.valid_writer = SummaryWriter(log_dir=os.path.join(self.exp_path, 'valid'))
        if USE_CUDA:
            self.model.cuda()
        self.max_acc = 0
        self.min_test_loss = 100
        train_samples = len(self.train_dataloader.dataset)
        self.display_step = self.calculate_display_step(num_sample=train_samples, batch_size=self.batch_size)
        self.checkpoint_saving_steps = checkpoint_saving_steps
        self.lr_scheduler = lr_scheduler
        self.patience = patience
        self.batch_transform = batch_transform
        self.batch_valid_transform = batch_valid_transform
        # self.mixup = self.train_dataloader.dataset.mixup
        # self.is_spec_transform = self.train_dataloader.dataset.is_data_augmentation
    

    def fit(self):
        for self.epoch in range(1, self.train_epoch + 1):
            self.train()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
                lr = self.lr_scheduler.get_last_lr()[0]
                print(f'learning rate {lr}')
                if self.USE_TENSORBOARD:
                    self.train_writer.add_scalar('Learning_rate', lr, self.epoch)
                    self.valid_writer.add_scalar('Learning_rate', lr, self.epoch)
            with torch.no_grad():
                if self.valid_dataloader is not None:
                    self.validate()
                    if self.early_stopping():
                        break
                self.save_model()
        # self.save_checkpoint('ckpt_final')

        if self.USE_TENSORBOARD:
            self.train_writer.close()
            self.valid_writer.close()
    
    def predict(self, input):
        # TODO: what is aux in model?
        output = self.model(input)
        # outputs = self.model(input_var)['out']
        return output

    def train(self):
        self.model.train()
        print(60*"=")
        self.logger.info(f'Epoch {self.epoch}/{self.train_epoch}')
        train_samples = len(self.train_dataloader.dataset)
        total_train_loss = 0.0
        
        for i, data in enumerate(self.train_dataloader, self.iterations + 1):
            input_var, target_var = data['input'], data['target']
            input_var, target_var = input_var.float(), target_var.float()
            input_var, target_var = input_var.to(self.device), target_var.to(self.device)
            if self.batch_transform is not None:
                input_var, target_var = self.batch_transform(input_var, target_var)

            batch_output = self.predict(input_var)
            
            self.optimizer.zero_grad()
            loss = self.criterion(batch_output, target_var)
            loss.backward()
            self.optimizer.step()
        
            loss = loss.item()
            total_train_loss += loss*input_var.shape[0]
            if self.USE_TENSORBOARD:
                self.train_writer.add_scalar('Loss/step', loss, i)

            if i%self.display_step == 0:
                self.logger.info('Step {}  Step loss {}'.format(i, loss))
        self.iterations = i
        train_loss = total_train_loss/train_samples
        self.logger.info(f'Training loss {train_loss}')
        if self.USE_TENSORBOARD:
            # TODO: correct total_train_loss
            self.train_writer.add_scalar('Loss/epoch', train_loss, self.epoch)

        mlflow.log_metric('train_loss', train_loss, step=self.epoch)

    def validate(self):
        self.model.eval()
        # self.eval_tool = metrics.SegmentationMetrics(self.n_class, ['accuracy'])
        test_n_iter, total_test_loss = 0, 0
        valid_samples = len(self.valid_dataloader.dataset)
        total_labels, total_preds = [], []
        # total_labels, total_preds = torch.tensor([]), torch.tensor([])
        # TODO: separate the acc part
        for idx, data in enumerate(self.valid_dataloader):
            test_n_iter += 1
            input_var, target_var = data['input'], data['target']
            input_var, target_var = input_var.float(), target_var.float()
            input_var, target_var = input_var.to(self.device), target_var.to(self.device)
            # XXX: mixup?
            if self.batch_valid_transform is not None:
                input_var, target_var = self.batch_valid_transform(input_var, target_var)

            output = self.predict(input_var)
            loss = self.criterion(output, target_var)

            # loss = loss_func(outputs, torch.argmax(target_var, dim=1)).item()
            total_test_loss += loss.item()

            prob = self.valid_activation(output)
            prediction = torch.argmax(prob, dim=1)
            # XXX
            target_var = target_var[:, 1]

            # total_labels = torch.cat((total_labels, target_var), 0)
            # total_preds = torch.cat((total_preds, prediction), 0)

            target_var = target_var.cpu().detach().numpy()
            prediction = prediction.cpu().detach().numpy()
            total_labels.append(np.reshape(target_var, target_var.size))
            total_preds.append(np.reshape(prediction, prediction.size))

        # TODO: convert to cpu after finish iteration
        # TODO: extend to multi-classes using
        total_labels = np.concatenate(total_labels)
        total_preds = np.concatenate(total_preds)
        self.avg_test_acc = accuracy_score(total_labels, total_preds)
        self.test_loss = total_test_loss/valid_samples
        self.logger.info(f'Testing loss {self.test_loss}')

        self.valid_writer.add_scalar('Accuracy/epoch', self.avg_test_acc, self.epoch)
        self.valid_writer.add_scalar('Loss/epoch', self.test_loss, self.epoch)

        mlflow.log_metric('valid_loss', self.test_loss, step=self.epoch)

    def early_stopping(self):
        if self.test_loss > self.min_test_loss:
            self.trigger += 1
            if self.trigger > self.patience:
                print(f'[Early stopping]  Epoch {self.epoch}')
                return True
        else:
            self.trigger = 0
            return False

    def load_model_from_checkpoint(self, ckpt, model_state_key='model_state_dict'):
        state_key = torch.load(ckpt, map_location=self.device)
        self.model.load_state_dict(state_key[model_state_key], strict=self.strict)
        self.model = self.model.to(self.device)

    def save_model(self):
        if self.test_loss < self.min_test_loss:
            self.min_test_loss = self.test_loss
            self.logger.info(f"-- Saving best model with testing loss {self.min_test_loss:.3f} --")
            checkpoint_name = 'ckpt_best'
            self.save_checkpoint(checkpoint_name)

            mlflow.log_metric('best_val_acc', self.avg_test_acc, step=self.epoch)
            
        # if self.epoch%self.config.TRAIN.CHECKPOINT_SAVING_STEPS == 0:
        if self.epoch%self.checkpoint_saving_steps == 0:
            self.logger.info(f"Saving model with testing accuracy {self.avg_test_acc:.3f} in epoch {self.epoch} ")
            checkpoint_name = f'ckpt_best_{self.epoch:04d}'
            self.save_checkpoint(checkpoint_name)

    def save_checkpoint(self, checkpoint_name):
        checkpoint = {
            "net": self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            "epoch": self.epoch
        }
        torch.save(checkpoint, os.path.join(self.exp_path, f'{checkpoint_name}.pth'))
    
    def calculate_display_step(self, num_sample, batch_size, display_times=5):
        num_steps = max(num_sample//batch_size, 1)
        display_steps = max(num_steps//display_times, 1)
        # display_steps = max(num_steps//display_times//display_times*display_times, 1)
        return display_steps


if __name__ == '__main__':
    pass