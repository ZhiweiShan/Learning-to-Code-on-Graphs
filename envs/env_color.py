import torch
import dgl
from collections import namedtuple
import dgl.function as fn
import networkx as nx
import numpy as np
from copy import deepcopy as dc
import random
import time
from time import time
from torch.utils.data import DataLoader


class MaximumIndependentSetEnv(object):
    def __init__(
            self,
            max_epi_t,
            max_num_nodes,
            hamming_reward_coef,
            device,
            num_color
    ):
        self.max_epi_t = max_epi_t
        self.max_num_nodes = max_num_nodes
        self.hamming_reward_coef = hamming_reward_coef
        self.device = device
        self.num_color = num_color

    def step(self, action):
        # print(action)
        reward, sol, done, in_color_reward = self._take_action(action)
        ob = self._build_ob()
        self.sol = sol
        info = {"sol": self.sol}

        return ob, reward, done, info, in_color_reward

    def _take_action(self, action):
        undecided = self.x == 0  # 找到defered 索引
        self.x[undecided] = action[undecided]  # 将defered node更新
        self.t += 1
        #        rg = self.g.reverse(share_ndata=True, share_edata=True).to(self.device)
        for i in range(1, self.num_color + 1):  # 对每一种color：
            x1 = (self.x == i)  # 标记是否为该种颜色 (True or False)
            self.g = self.g.to(self.device)
            self.g.ndata['h'] = x1.float()  # False -> 0, True -> 1
            self.g.update_all(
                fn.copy_src(src='h', out='m'),
                fn.sum(msg='m', out='h')
            )
            # 通过消息传递求degree
            x1_deg = self.g.ndata.pop('h')  # 将ndata存进x1_deg并删除'h'

            # 找出冲突的点
            clashed = x1 & (x1_deg > 0)
            self.x[clashed] = 0  # 将冲突的node打回0

        # fill timeout with zeros
        still_undecided = (self.x == 0)
        timeout = (self.t == self.max_epi_t)
        self.x[still_undecided & timeout] = self.num_color + 1  # 超时的多加一个颜色

        done = self._check_done()
        self.epi_t[~done] += 1

        # compute reward and solution
        x_not_0 = (self.x != 0).float()
        x_not_new_color = (self.x != self.num_color + 1).float()
        x_colored = (x_not_0 + x_not_new_color == 2).float()  # 即不是0，也不是新color时给一个reward
        # x_colored = x_not_0
        h = x_colored
        self.g.ndata['h'] = h
        next_sol = dgl.sum_nodes(self.g, 'h')
        self.g.ndata.pop('h')

        reward = (next_sol - self.sol)

        reward /= self.max_num_nodes
        reward[done & (self.already_done == 0)] += (self.max_epi_t - self.t[0][0]).float() / self.max_epi_t
        self.already_done[done] = 1

        return reward, next_sol, done, 0

    def _check_done(self):
        undecided = (self.x == 0).float()
        self.g.ndata['h'] = undecided
        num_undecided = dgl.sum_nodes(self.g, 'h')
        self.g.ndata.pop('h')
        done = (num_undecided == 0)

        return done

    def _build_ob(self):
        ob_x = self.x.unsqueeze(2).float()
        ob_t = self.t.unsqueeze(2).float() / self.max_epi_t
        ob = torch.cat([ob_x, ob_t], dim=2)
        return ob

    def register(self, g, num_samples=1):
        self.g = g
        self.num_samples = num_samples
        self.g.set_n_initializer(dgl.init.zero_initializer)
        self.g.to(self.device)
        self.batch_num_nodes = torch.LongTensor(
            self.g.batch_num_nodes()
        ).to(self.device)

        num_nodes = self.g.number_of_nodes()
        self.x = torch.full(
            (num_nodes, num_samples),
            0,  # 原为2
            dtype=torch.long,
            device=self.device
        )
        self.t = torch.zeros(
            num_nodes,
            num_samples,
            dtype=torch.long,
            device=self.device
        )

        ob = self._build_ob()

        self.sol = torch.zeros(
            self.g.batch_size,
            num_samples,
            device=self.device
        )
        self.num_color_used = self.num_color* torch.ones(
            self.g.batch_size,
            num_samples,
            device=self.device
        )
        self.epi_t = torch.zeros(
            self.g.batch_size,
            num_samples,
            device=self.device
        )
        self.already_done = torch.zeros(
            self.g.batch_size,
            num_samples,
            device=self.device
        )

        return ob