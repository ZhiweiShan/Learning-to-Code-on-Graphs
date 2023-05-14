import torch
import dgl
import dgl.function as fn
import numpy as np
from numpy.linalg import matrix_rank as rk


class MaximumIndependentSetEnv(object):
    def __init__(
        self, 
        max_epi_t, 
        max_num_nodes, 
        hamming_reward_coef, 
        device,
        num_color,
        local_give_up_coef,
        space_dim
        ):
        self.max_epi_t = max_epi_t
        self.max_num_nodes = max_num_nodes
        self.hamming_reward_coef = hamming_reward_coef
        self.device = device
        self.num_color = num_color
        self.local_give_up_coef = local_give_up_coef
        self.space_dim = space_dim
        
    def step(self, action):
        #print(action)
        reward, sol, done, max_num_in_color = self._take_action(action)
        
        ob = self._build_ob()
        self.sol = sol
        info = {"sol": self.sol}

        return ob, reward, done, info, max_num_in_color


    def send_source(self,edges):
        return {'m': edges.src['h']}


    def count_rank(self,nodes):
        def rk(mailbox):
            T = np.array(mailbox.permute(0,2,1,3).cpu())
            rank = torch.tensor(np.linalg.matrix_rank(T)).to(self.device)
            return rank

        return {'h': rk(nodes.mailbox['m'])}

    def cat(self,nodes):
        def rk(mailbox):
            T = mailbox
            T = T.reshape([T.size()[0],1,T.size()[2],-1])
            return T

        return {'h': rk(nodes.mailbox['m'])}

    def _take_action(self, action):

        undecided = self.x == 0
        self.x[undecided] = action[undecided]
        self.t += 1

        # Clean:
        x1 = self.vecs[self.x]

        self.g = self.g.to(self.device)
        self.g.ndata['h'] = x1.float()
        self.g.update_all(
            self.send_source,
            self.count_rank
            )
        rk_N = self.g.ndata.pop('h')


        self.g_selfloop.ndata['h'] = x1.float()
        self.g_selfloop.update_all(
            self.send_source,
            self.count_rank
            )
        rk_NS = self.g_selfloop.ndata.pop('h')


        clash1 = rk_N == self.space_dim
        x0 = torch.count_nonzero(x1, dim=2) == 0
        clash2 = rk_N == rk_NS
        clashed = clash1 | (clash2 & ~x0)

        self.rg.ndata['h'] = clashed.float()
        self.rg.update_all(
            fn.copy_src(src='h', out='m'),
            fn.sum(msg='m', out='h')
        )
        clashed_neighbor = self.rg.ndata.pop('h')
        clashed_set = clashed | (clashed_neighbor > 0)

        self.x[clashed_set] = 0


        # fill timeout with zeros
        still_undecided = (self.x == 0)
        timeout = (self.t == self.max_epi_t)
        self.x[still_undecided & timeout] = self.num_color+1

        done = self._check_done()
        self.epi_t[~done] += 1

        # compute reward and solution
        x_not_0 = (self.x != 0 ).float()
        x_not_new_color = (self.x != self.num_color+1).float()
        x_colored = (x_not_0 + x_not_new_color == 2).float()
        #x_colored = x_not_0
        h = x_colored
        self.g.ndata['h'] = h
        next_sol = dgl.sum_nodes(self.g, 'h')
        self.g.ndata.pop('h')

        reward = (next_sol - self.sol)



        reward /= self.max_num_nodes
        reward[done & (self.already_done == 0)] += (self.max_epi_t - self.t[0][0]).float() / self.max_epi_t
        self.already_done[done] = 1
        max_num_in_color = 0
        return reward, next_sol, done, max_num_in_color



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
        ob = torch.cat([ob_x, ob_t], dim = 2)
        return ob
        
    def register(self, g, num_samples = 1):
        self.g = g
        self.g = self.g.to(self.device)
        self.rg = self.g.reverse(copy_ndata=True, copy_edata=True).to(self.device)
        self.g_selfloop = self.g.add_self_loop().to(self.device)
        self.num_samples = num_samples
        self.g.set_n_initializer(dgl.init.zero_initializer)
        self.g.to(self.device)
        self.batch_num_nodes = self.g.batch_num_nodes()
        
        num_nodes = self.g.number_of_nodes()
        self.x = torch.full(
            (num_nodes, num_samples),
            0,
            dtype = torch.long, 
            device = self.device
            )
        self.x_vector = torch.full(
            (num_nodes, self.space_dim, num_samples),
            0,
            dtype = torch.long,
            device = self.device
            )
        self.t = torch.zeros(
            num_nodes, 
            num_samples, 
            dtype = torch.long, 
            device = self.device
            )
        
        ob = self._build_ob()
        
        self.sol = torch.zeros(
            self.g.batch_size, 
            num_samples, 
            device = self.device
            )
        self.epi_t = torch.zeros(
            self.g.batch_size, 
            num_samples, 
            device = self.device
            )            
        self.already_done = torch.zeros(
            self.g.batch_size,
            num_samples,
            device=self.device
        )
        self.vecs = torch.zeros(2 ** self.space_dim, self.space_dim, device=self.device)
        for i in range(2 ** self.space_dim):
            bin_str = format(i, f'0{self.space_dim}b')
            vec = torch.tensor([int(b) for b in bin_str])
            self.vecs[i] = vec

        return ob