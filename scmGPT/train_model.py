from mingpt.utils import CfgNode as CN
import torch
from torch.utils.data import (DataLoader, Dataset)
from tqdm.auto import tqdm
import numpy as np

class Trainer:

    @staticmethod
    def get_default_config():
        C = CN()
        # device to train on
        C.device = 'auto'
        # dataloder parameters
        C.num_workers = 1
        # optimizer parameters
        C.epoch = 100
        C.batch_size = 64
        C.learning_rate = 3e-4
        C.betas = (0.95, 0.99)
        C.weight_decay = 0.1  # only applied on matmul weights
        C.grad_norm_clip = 1.0
        return C

    def __init__(self, config, model, train_dataset):
        self.config = config
        self.model = model
        self.optimizer = None
        self.train_dataset = train_dataset
        # determine the device we'll train on
        if config.device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = config.device
        self.model = self.model.to(self.device)
        print("running on device", self.device)

        # variables that will be assigned to trainer class later for logging and etc
        self.iter_num = 0
        self.iter_time = 0.0
        self.iter_dt = 0.0

    def run(self):
        model, config = self.model, self.config

        # setup the optimizer
        self.optimizer = model.configure_optimizers(config)

        train_loader = DataLoader(self.train_dataset, batch_size=config.batch_size, shuffle=False,
                                  pin_memory=True)

        model.train()
        n_epochs = config.epoch
        for epoch in range(n_epochs):

            train_loss = []
            train_loss1 = []
            train_loss3 = []
            emb_mod1s = []
            emb_mod2s = []
            mod1_logits2s = []
            for batch in tqdm(train_loader):
                X, Y, = batch
                X = X.to(self.device)
                Y = Y.to(self.device)
                emb_mod1, emb_mod2, self.loss,self.loss1,self.loss3, mod1_logits2 = model(X, Y)
                model.zero_grad(set_to_none=True)
                self.loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
                self.optimizer.step()
                train_loss.append(self.loss.item())
                train_loss1.append(self.loss1.item())
                train_loss3.append(self.loss3.item())

                if(self.device == 'cuda'):
                    emb_mod1s.extend(emb_mod1.cpu().detach().numpy()) #.numpy()
                    emb_mod2s.extend(emb_mod2.cpu().detach().numpy()) #.numpy()
                    mod1_logits2s.extend(mod1_logits2.cpu().detach().numpy()) #.numpy()
                else:
                    emb_mod1s.extend(emb_mod1.detach().numpy()) #.numpy()
                    emb_mod2s.extend(emb_mod2.detach().numpy()) #.numpy()
                    mod1_logits2s.extend(mod1_logits2.detach().numpy()) #.numpy()

            train_loss = sum(train_loss) / len(train_loss)
            train_loss1 = sum(train_loss1) / len(train_loss1)
            train_loss3 = sum(train_loss3) / len(train_loss3)

            print(f"[ Train | {epoch + 1:03d}/{n_epochs:03d} ] loss = {train_loss:.5f},"
                  f"loss1 = {train_loss1:.5f},"
                  f"loss3 = {train_loss3:.5f},")

        emb_mod1s = np.asarray(emb_mod1s)
        emb_mod2s = np.asarray(emb_mod2s)

        mod1_logits2s = np.asarray(mod1_logits2s)

        return emb_mod1s, emb_mod2s, mod1_logits2s#, mod2_logits1s
