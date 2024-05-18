import torch
import dgl
import dgl.function as fn
import numpy as np


class MaximumIndependentSetEnv(object):
    def __init__(
        self, 
        max_epi_t, 
        max_num_nodes, 
        device,
        num_color,
        local_give_up_coef,
        clean2,
        num_antenna
        ):
        self.max_epi_t = max_epi_t
        self.max_num_nodes = max_num_nodes
        self.device = device
        self.num_color = num_color
        self.local_give_up_coef = local_give_up_coef
        self.clean2 = clean2
        self.num_antenna = num_antenna
        
    def step(self, action):
        reward, sol, done, max_local_color = self._take_action(action)
        
        ob = self._build_ob()
        self.sol = sol
        info = {"sol": self.sol}

        return ob, reward, done, info, max_local_color



    def send_source(self,edges):
        return {'m': edges.src['h']}

    # def count_reduce(self,nodes):
    #     def count_color(mailbox):
    #         color_count = torch.ones((mailbox.shape[0], 1, mailbox.shape[2]), dtype=int).to(self.device)
    #         if (mailbox.shape[1] >= self.num_color - self.clean2 and mailbox.shape[1] != 1):
    #             T = np.array(mailbox.cpu())
    #             color_count = torch.tensor(np.apply_along_axis(lambda row:
    #             [len(np.delete(np.unique(row),np.where(np.unique(row)==0)))], axis=1, arr=T), dtype=torch.int64).to(self.device)
    #         return color_count.squeeze(dim=1)
    #
    #     return {'c': count_color(nodes.mailbox['m'])}

    def record_reduce(self, nodes):
        def count_color(mailbox):
            num_rows = mailbox.size(0)
            num_cols = mailbox.size(2)
            color_tensor = torch.zeros(num_rows, num_cols, self.num_color + 1, dtype=torch.int64, device=mailbox.device)
            for idx, row in enumerate(mailbox.permute(0, 2, 1)):
                for i in range(self.num_color + 1):
                    color_tensor[idx, :, i] += (row == i).any(dim=1)
            return color_tensor


        return {'c': count_color(nodes.mailbox['m'])}
    def _take_action(self, action):

        undecided = self.x == 0
        self.x[undecided] = action[undecided]
        self.t += 1
        rg = self.g.reverse(copy_ndata=True, copy_edata=True).to(self.device)
        # Clean 1:
        for i in range(1,self.num_color+1):
            x1 = (self.x == i)
            self.g = self.g.to(self.device)
            self.g.ndata['h'] = x1.float()
            self.g.update_all(
                fn.copy_src(src='h', out='m'),
                fn.sum(msg='m', out='h')
                )

            x1_deg = self.g.ndata.pop('h')

            clashed = x1 & (x1_deg > self.num_antenna-1) #for SIMO with number of antennas (1,n), clean x1_deg > n-1


            rg.ndata['h'] = clashed.float()
            rg.update_all(
                fn.copy_src(src='h', out='m'),
                fn.sum(msg='m', out='h')
            )
            x_clashed_deg = rg.ndata.pop('h')
            clashed_2 = x1 & (x_clashed_deg > 0)

            self.x[clashed] = 0
            self.x[clashed_2] = 0

        # clean1 finished，next clean2
        if self.t[0][0] < self.max_epi_t*self.local_give_up_coef:
            self.g.ndata['h'] = self.x
            self.g.update_all(
                self.send_source,
                self.record_reduce)
            # neighbor's color
            color_tensor = self.g.ndata.pop('c').float()

            # add self
            for i in range(self.num_color+1):
                indices = torch.where(self.x == i)
                color_tensor[indices[0], indices[1], i ] = 1

            # count number (except 0)
            num_local_color = torch.sum(color_tensor[:, :, 1:], dim=2)

            x_reach_C = (num_local_color >= self.num_color-self.clean2 )
            rg.ndata['h'] = x_reach_C.float()
            rg.update_all(
                fn.copy_src(src='h', out='m'),
                fn.sum(msg='m', out='h')
            )
            reach_C_neighbor = rg.ndata.pop('h')
            reach_C_set = x_reach_C | (reach_C_neighbor > 0)

            self.x[reach_C_set] = 0


        # fill timeout with self.num_color+1
        still_undecided = (self.x == 0)
        timeout = (self.t == self.max_epi_t)
        self.x[still_undecided & timeout] = self.num_color+1

        done = self._check_done()
        self.epi_t[~done] += 1

        # compute reward and solution
        x_not_0 = (self.x != 0 ).float()
        x_not_new_color = (self.x != self.num_color+1).float()
        x_colored = (x_not_0 + x_not_new_color == 2).float()

        h = x_colored
        self.g.ndata['h'] = h
        next_sol = dgl.sum_nodes(self.g, 'h')
        self.g.ndata.pop('h')

        reward = (next_sol - self.sol)


        max_local_color = torch.tensor(0.)

        if torch.all(done).item():
            # self.g.ndata['h'] = self.x
            # self.g.update_all(
            #     self.send_source,
            #     self.count_reduce
            #     )
            # num_in_color = self.g.ndata.pop('c').float()

            self.g.ndata['h'] = self.x
            self.g.update_all(
                self.send_source,
                self.record_reduce)
            # neighbor's color
            color_tensor = self.g.ndata.pop('c').float()

            # add self
            for i in range(self.num_color+1):
                indices = torch.where(self.x == i)
                color_tensor[indices[0], indices[1], i ] = 1

            # count number (except 0)
            num_local_color = torch.sum(color_tensor[:, :, 1:], dim=2)
            self.g.ndata.pop('h')
            self.g.ndata['h'] = num_local_color
            max_local_color = dgl.max_nodes(self.g, 'h')
            self.g.ndata.pop('h')


        reward /= self.max_num_nodes
        reward[done & (self.already_done == 0)] += (self.max_epi_t - self.t[0][0]).float() / self.max_epi_t
        self.already_done[done] = 1
        return reward, next_sol, done, max_local_color



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
        self.num_samples = num_samples
        self.g.set_n_initializer(dgl.init.zero_initializer)
        self.g.to(self.device)
        self.batch_num_nodes = torch.LongTensor(
            self.g.batch_num_nodes()
            ).to(self.device)
        
        num_nodes = self.g.number_of_nodes()
        self.x = torch.full(
            (num_nodes, num_samples),
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
        return ob