r'''
Module to run a GAN architecture to generate text.
'''
# standard library imports
from __future__ import  absolute_import
import math
import tqdm
import random
import shutil
from os import getcwd
from datetime import datetime

# third party imports
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.backends import cudnn
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
from termcolor import cprint

# internal imports
from agents.base_agent import BaseAgent
from models.gantext.generator import Generator
from models.gantext.discriminator import Discriminator
from dataloaders.wikitext2_dl import WikiText2DL
from dataloaders.navinstr_dl import NavigationInstructionsDL
from metrics.bleu import BLEU
from metrics.perplexity import Perplexity
from utils.utils import debug_memory, show_gpu, display_status, display_state_dict, print_line, save_experiment


cudnn.benchmark = True


class GANTextAgent(BaseAgent):
    '''
    Class that implements BaseAgent to train
    a discriminator and generator using a GAN architecture
    to create short examples of text.

    Arguments
    ---------
        config : obj
            Configuration object.
    '''

    def __init__(self, config):
        ''' Initializes an instance of the GANTextAgent class. '''
        super().__init__(config)

        self.start = None
        self.end = None
        self.save_filepath = getcwd() + self.config.checkpoint_dir

        self.cuda_avail = torch.cuda.is_available()
        if self.cuda_avail and not self.config.cuda:
            self.logger.info('WARNING :: CUDA is available but is not currently being used')
        self.cuda = self.cuda_avail and self.config.cuda

        self.manual_seed = self.config.seed
        if self.manual_seed <= 0:
            self.manual_seed = random.randint(1, 10000)
            self.logger.info('INFO :: Random seed used.')
        self.logger.info('SEED :: ', self.manual_seed)
        random.seed(self.manual_seed)
        
        if self.cuda:
            self.device = torch.device('cuda')
            torch.cuda.set_device(self.config.gpu_device)
            torch.manual_seed(self.manual_seed)
            torch.cuda.manual_seed(self.manual_seed)
            self.logger.info('CUDA in use')
            show_gpu()
        else:
            self.device = torch.device('cpu')
            torch.manual_seed(self.manual_seed)
            self.logger.info('CPU in use (not on CUDA)')

        # data loading
        self.dataloader_class = globals()[config.dataloader]
        self.dataloader = self.dataloader_class(config=self.config, device=self.device)
        self.data_dict = self.dataloader.get_data()
        self.train_loader = self.data_dict.train_loader
        self.valid_loader = self.data_dict.valid_loader
        self.train_iter = self.data_dict.train_iter
        self.vocab_size = self.data_dict.vocab_size
        self.vocab = self.data_dict.vocab

        # set epoch and batch number tracking
        self.current_epoch = 0
        self.num_epochs = self.config.num_epochs
        self.current_batch = 0

        self.Generator = globals()[config.generator]
        self.Discriminator = globals()[config.discriminator]

        # define the networks (G and D) and their optimizers
        self.G = Generator(
            config=self.config,
            vocab_size=self.vocab_size,
            _device=self.device,
            _cuda=self.cuda
        )
        self.D = Discriminator(
            self.config,
            vocab_size=self.vocab_size,
            _device=self.device,
            _cuda=self.cuda
        )
        self.G_optim = optim.AdamW(
            self.G.parameters(),
            lr=float(self.config.learning_rate),
            betas=(self.config.beta1, self.config.beta2)
        )
        self.D_optim = optim.AdamW(
            self.D.parameters(),
            lr=float(self.config.learning_rate),
            betas=(self.config.beta1, self.config.beta2)
        )

        self.criterion = nn.BCELoss()
        
        # send G, D, loss, and fixed noise to GPU or CPU (depending on config)
        self.G = self.G.to(self.device)
        self.D = self.D.to(self.device)
        self.criterion = self.criterion.to(self.device)
        self.fixed_noise = Variable(torch.randn(self.config.batch_size, self.config.g_input_size, 1, 1))
        self.fixed_noise = self.fixed_noise.to(self.device)

        # define the real and fake label
        self.real_label = 1
        self.fake_label = 0

        # load the checkpoint if the config file points to one to load.
        try:
            self.load_checkpoint(self.config.checkpoint_file)
        except AttributeError:
            pass
        
        # checkpoint file for saving models
        self.checkpoint_file = self.config.checkpoint_file
        self.is_best = False

        # tensorboard
        self.writer = SummaryWriter(
            log_dir=getcwd() + self.config.log_dir,
            comment='GANText'
        )

    def load_checkpoint(self, file_name='gantext.pth'):
        ''' Loads the checkpoint from the given file. '''
        filepath = self.config.checkpoint_dir + file_name

        try:
            self.logger.info(f'Loading checkpoint from {filepath}')
            checkpoint = torch.load(filepath)

            self.current_epoch = checkpoint['epoch']
            self.current_batch = checkpoint['batch']
            self.G.load_state_dict(checkpoint['G_state_dict'])
            self.D.load_state_dict(checkpoint['D_state_dict'])
            self.G_optim.load_state_dict(checkpoint['G_optim'])
            self.D_optim.load_state_dict(checkpoint['D_optim'])
            self.fixed_noise = checkpoint['fixed_noise']
            self.manual_seed = checkpoint['manual_seed']

            self.logger.info(f'Checkpoint loaded from {filepath}\t\t::\t\t\
                Currently at batch {self.current_batch} of epoch {self.current_epoch}.')
        except OSError:
            self.logger.info(f'No checkpoint at the specifed file path {filepath}.\
                Skipping for now.')
            self.logger.info('Checkpoint load failure most likely because filepath is incorrect \
                or this is the first training run.')
    
    def save_checkpoint(self, file_name='gantext.pth', is_best=False):
        '''
        Saves the models to the given file.

        Arguments
        ---------
            file_name : str             (default='checkpoint.pth')
                Name of the file to save model to.
            is_best : bool
                If true, this is the best model to date.
        '''
        state = {
            'epoch': self.current_epoch,
            'batch': self.current_batch,
            'G_state_dict': self.G.state_dict(),
            'D_state_dict': self.D.state_dict(),
            'G_optim': self.G_optim.state_dict(),
            'D_optim': self.D_optim.state_dict(),
            'fixed_noise': self.fixed_noise,
            'manual_seed': self.manual_seed
        }

        self.save_filepath = getcwd() + self.config.checkpoint_dir + file_name
        while True:
            try:
                torch.save(state, self.save_filepath)
                break
            except FileNotFoundError:
                cprint('ERROR :: File not found.', 'red')
                cprint('INFO :: Creating file and trying to checkpoint again', 'yellow')
                f = open(self.save_filepath, 'w+')
        if is_best:
            shutil.copyfile(self.save_filepath, self.config.checkpoint_dir + 'best_model.pth')
    
    def flip_labels(self):
        ''' Flips real and fake labels. '''
        if self.real_label == 1:
            self.real_label = 0
            self.fake_label = 1
        else:
            self.real_label = 1
            self.fake_label = 0
    
    def run(self):
        ''' Runs the GAN architecture. '''
        try:
            losses = self.train()
            return losses
        except Exception as e:
            raise e
    
    def train(self):
        ''' Function to train the discriminator and generator in sequence. '''
        losses = 0
        self.start = datetime.now()
        for epoch in range(self.current_epoch, self.config.num_epochs):
            self.current_epoch = epoch
            self.train_one_epoch()
            self.save_checkpoint(file_name=self.checkpoint_file, is_best=False)
            if self.current_epoch == math.floor(self.config.num_epochs / 2):
                self.flip_labels()
        self.end = datetime.now()
        return losses
    
    def train_one_epoch(self):
        ''' Runs a single training epoch. '''
        self.G.train()
        self.D.train()
        for x in self.train_loader:
            print(x)
            # D REAL TRAINING
            self.D.zero_grad()
            x = x.type(torch.FloatTensor)
            real = torch.Tensor(x)
            if self.config.bidirectional:
                target = torch.ones(2 * self.config.num_layers * 2, device=self.device)
            else:
                target = torch.ones(2 * self.config.num_layers, device=self.device)
            real = real.type(torch.LongTensor)
            if self.cuda:
                real = real.to(self.device)
                target = target.to(self.device)
            output = self.D(real)
            output = output.view(-1)
            # calculate loss
            errD_real = self.criterion(output, target)
            errD_real.backward()
            D_x = output.mean().item()

            # D FAKE TRAINING
            real = real.type(torch.FloatTensor)
            noise = torch.rand_like(real)
            noise = noise.type(torch.LongTensor)
            if self.cuda:
                noise = noise.to(self.device)
            # generate fake data
            fake, hidden = self.G(noise)
            target.fill_(self.fake_label)
            fake = fake.detach()
            fake = fake.type(torch.LongTensor)
            if self.cuda:
                fake = fake.to(self.device)
            output = self.D(fake)
            # calculate loss
            errD_fake = self.criterion(output, target)
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            errD = errD_real + errD_fake
            self.D_optim.step()

            # G TRAINING
            self.G.zero_grad()
            target.fill_(self.real_label)
            output = self.D(fake)
            errG = self.criterion(output, target)
            errG.backward()
            D_G_z2 = output.mean().item()
            self.G_optim.step()

            # display status of training every 100 batches
            if self.current_batch % 100 == 0:
                display_status(
                    self.current_epoch + 1, self.num_epochs,
                    self.current_batch + 1, len(self.train_loader),
                    errD.item(),
                    errG.item(),
                    D_x, D_G_z1, D_G_z2
                )
            
            self.current_batch += 1
        
        # reset current_batch to 0 after epoch is finished
        self.current_batch = 0

    def validate(self):
        ''' Runs a cycle of validation on the model(s). Not implemented for now. '''
        pass
    
    def finalize(self):
        ''' Finalizes training (outputs graphs, etc.). '''
        # calculate bleu and perplexity
        # bleu, perplexity = self.evaluate()
        self.load_checkpoint(file_name='checkpoint.pth')

        if self.is_best:
            self.save_checkpoint(file_name=self.checkpoint_file, is_best=True)

        # display the state dicts of each 
        display_state_dict(
            model=self.G,
            model_name='Generator',
            optimizer=self.G_optim
        )
        display_state_dict(
            model=self.D,
            model_name='Discriminator',
            optimizer=self.D_optim
        )
        # self.writer.add_graph(model=self.G, input_to_model=self.train_loader, verbose=True)
        # self.writer.add_graph(model=self.D, input_to_model=self.train_loader, verbose=True)
        self.writer.close()
        print_line(color='green')
        cprint('FINISHED TRAINING', 'green')
        print_line(color='green')
        # print(self.sample(model=self.G, num_samples=10))
        print(self.G.predict(
            words=['walk', 'straight', 'green', 'room'],
            vocab=self.vocab, num_samples=10, topk=5)
        )
        cprint('EVALUATION METRICS', 'blue')
        print_line()
        duration = self.end - self.start
        save_experiment(
            checkpoint_path=self.save_filepath, config=self.config,
            seed=self.manual_seed, duration=duration, start_time=self.start, end_time=self.end
        )

    def evaluate(self):
        ''' Evaluates the generator using various metrics. '''
        bleu = BLEU(model=self.G)
        perplexity = Perplexity(model=self.G)
        return bleu, perplexity