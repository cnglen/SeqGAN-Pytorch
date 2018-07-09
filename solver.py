#!/usr/bin/env python
# -*- coding: utf-8 -*-


import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from params import GenParams, DisParams
import util
import discriminator
import generator
import target_lstm
from util import GenData, DisData
from torch.utils.data import DataLoader
from rollout import Rollout
import time


class Solver(object):
    """docstring for Solver"""

    def __init__(self, vocab_size, batch_size, pre_gen_epochs, pre_dis_epochs, gan_epochs, generate_sum, sequence_len, lr, real_file, fake_file, eval_file, update_rate):
        super(Solver, self).__init__()
        self.vocal_size = vocab_size
        self.batch_size = batch_size
        self.pre_gen_epochs = pre_gen_epochs
        self.pre_dis_epochs = pre_dis_epochs
        self.gan_epochs = gan_epochs
        self.generate_sum = generate_sum
        self.sequence_len = sequence_len
        self.lr = lr
        self.real_file = real_file
        self.fake_file = fake_file
        self.eval_file = eval_file
        self.update_rate = update_rate

        self.discriminator = discriminator.Discriminator(sequence_len, vocab_size, DisParams.emb_dim,
                                                         DisParams.filter_sizes, DisParams.num_filters,
                                                         DisParams.dropout)
        self.generator = generator.Generator(vocab_size, GenParams.emb_dim, GenParams.hidden_dim, GenParams.num_layers)
        self.target_lstm = target_lstm.TargetLSTM(
            vocab_size, GenParams.emb_dim, GenParams.hidden_dim, GenParams.num_layers)

        self.discriminator = util.to_cuda(self.discriminator)
        self.generator = util.to_cuda(self.generator)
        self.target_lstm = util.to_cuda(self.target_lstm)

    def train_epoch(self, model, data_loader, criterion, optim):
        """
        train the `model` using the data in `data_loader` for one epoch

        Return:
          loss
        """
        total_loss = 0.
        total_words = 0.
        for i, (data, target) in enumerate(data_loader):
            optim.zero_grad()
            # x: (None, sequence_len + 1), y: (None, sequence_len + 1)
            x, y = util.to_var(data), util.to_var(target)
            logits = model(x)  # (None, vocal_size, sequence_len+1)
            loss = criterion(logits, y)
            total_loss += loss.data.cpu().item()
            total_words += x.size(0) * x.size(1)
            loss.backward()
            optim.step()
        return total_loss / total_words

    def eval_epoch(self, model, data_loader, criterion):
        """
        evaluate the model using the data in `data_loader`

        Return:
           loss
        """
        total_loss = 0.
        total_words = 0.
        for i, (data, target) in enumerate(data_loader):
            # x: (None, sequence_len + 1), y: (None, sequence_len + 1). Should use volatile if no backward operation
            x, y = util.to_var(data, volatile=True), util.to_var(target, volatile=True)
            logits = model(x)  # (None, vocab_size, sequence_len+1)
            loss = criterion(logits, y)
            total_loss += loss.data.cpu().item()
            total_words += x.size(0) * x.size(1)
        return total_loss / total_words

    def pretrain_gen(self):
        """
        pretrain the generator
        """
        util.generate_samples(self.target_lstm, self.batch_size, self.sequence_len, self.generate_sum, self.real_file)
        gen_data = GenData(self.real_file)
        gen_data_loader = DataLoader(gen_data, batch_size=self.batch_size, shuffle=True, num_workers=8)
        gen_criterion = util.to_cuda(nn.CrossEntropyLoss(size_average=False, reduce=True))
        gen_optim = torch.optim.Adam(self.generator.parameters(), self.lr)
        print('\nPretrain generator......')
        for epoch in range(self.pre_gen_epochs):
            train_loss = self.train_epoch(self.generator, gen_data_loader, gen_criterion, gen_optim)
            util.generate_samples(self.generator, self.batch_size, self.sequence_len, self.generate_sum, self.eval_file)
            eval_data = GenData(self.eval_file)
            eval_data_loader = DataLoader(eval_data, batch_size=self.batch_size, shuffle=True, num_workers=8)
            eval_loss = self.eval_epoch(self.target_lstm, eval_data_loader, gen_criterion)
            print('epoch: {:4d}, train_loss: {:6.4f}, eval_loss: {:6.4f}'.format(epoch, train_loss, eval_loss))

    def pretrain_dis(self):
        """
        pretrain the discriminator
        """
        dis_criterion = util.to_cuda(nn.BCEWithLogitsLoss(size_average=False))
        dis_optim = torch.optim.Adam(self.discriminator.parameters(), self.lr)
        print('\nPretrain discriminator......')
        for epoch in range(self.pre_dis_epochs):
            util.generate_samples(self.generator, self.batch_size, self.sequence_len, self.generate_sum, self.fake_file)
            dis_data = DisData(self.real_file, self.fake_file)
            dis_data_loader = DataLoader(dis_data, batch_size=self.batch_size, shuffle=True, num_workers=8)
            loss = self.train_epoch(self.discriminator, dis_data_loader, dis_criterion, dis_optim)
            print('epoch: {:4d}, train_loss: {:6.4f}'.format(epoch, loss))

    def train_gan(self, backend):
        """
        train the GAN
        """

        rollout = Rollout(self.generator, self.discriminator, self.update_rate)
        print('\nStart Adeversatial Training......')
        gen_optim, dis_optim = torch.optim.Adam(self.generator.parameters(
        ), self.lr), torch.optim.Adam(self.discriminator.parameters(), self.lr)
        dis_criterion = util.to_cuda(nn.BCEWithLogitsLoss(size_average=False))
        gen_criterion = util.to_cuda(nn.CrossEntropyLoss(size_average=False, reduce=True))

        for epoch in range(self.gan_epochs):

            start = time.time()
            for _ in range(1):
                samples = self.generator.sample(self.batch_size, self.sequence_len)   # (batch_size, sequence_len)
                zeros = util.to_var(torch.zeros(self.batch_size, 1).long())   # (batch_size, 1)
                inputs = torch.cat([samples, zeros], dim=1)[:, :-1]  # (batch_size, sequence_len)
                # (batch_size, sequence_len)
                rewards = rollout.reward(samples, 16)
                rewards = util.to_var(torch.from_numpy(rewards))
                # (None, vocab_size, sequence_len)
                logits = self.generator(inputs)
                pg_loss = self.pg_loss(logits, samples, rewards)
                gen_optim.zero_grad()
                pg_loss.backward()
                gen_optim.step()

            print('  generator updated via policy gradient......')

            if epoch % 10 == 0:
                util.generate_samples(self.generator, self.batch_size,
                                      self.sequence_len, self.generate_sum, self.eval_file)
                eval_data = GenData(self.eval_file)
                eval_data_loader = DataLoader(eval_data, batch_size=self.batch_size, shuffle=True, num_workers=8)
                loss = self.eval_epoch(self.target_lstm, eval_data_loader, gen_criterion)
                print('  epoch: {0:d}, val loss: [{1:.4f}]'.format(epoch, loss))

            for _ in range(1):
                util.generate_samples(self.generator, self.batch_size, self.sequence_len,
                                      self.generate_sum, self.fake_file)
                dis_data = DisData(self.real_file, self.fake_file)
                dis_data_loader = DataLoader(dis_data, batch_size=self.batch_size, shuffle=True, num_workers=8)
                for _ in range(1):
                    loss = self.train_epoch(self.discriminator, dis_data_loader, dis_criterion, dis_optim)

            print('  discriminator updated via gan loss......')

            rollout.update_params()

            end = time.time()

            print('  speed: {:.3f}s/epoch in {}'.format(end - start, backend))

    def pg_loss(self, logits, actions, rewards):
        """
        logits: (None, vocab_size, sequence_len)
        actions: (None, sequence_len)
        rewards: (None, sequence_len)
        """
        neg_lik = F.cross_entropy(logits, actions, size_average=False, reduce=False)   # (None, sequence_len)
        loss = torch.mean(neg_lik * rewards)
        return loss
