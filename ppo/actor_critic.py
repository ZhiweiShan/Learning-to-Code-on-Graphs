import torch
import torch.nn as nn
from torch.distributions import Categorical
import networkx as nx
import dgl.function as fn
import dgl


class ActorCritic(nn.Module):
    def __init__(
            self,
            actor_class,
            critic_class,
            max_num_nodes,
            hidden_dim,
            num_layers,
            device,
            num_color
    ):
        super(ActorCritic, self).__init__()
        self.num_color = num_color
        self.output_dim = num_color + 1
        self.actor_net = actor_class(2 * num_color + 2, hidden_dim, self.output_dim,
                                     num_layers)  # (2, hidden_dim, 3, num_layers)
        self.critic_net = critic_class(2 * num_color + 2, hidden_dim, 1, num_layers)
        self.device = device
        self = self.to(device)
        self.max_num_nodes = max_num_nodes

    def get_masks_idxs_subg_h(self, ob, g):
        #   mask the deffered nodes
        # num_nodes x batch_size
        node_mask = (ob.select(2, 0).long() == 0)  # defered node -> node_mask=True #was == 2 #node_mask = (ob.select(2, 0).long() == 0)
        flatten_node_idxs = node_mask.reshape(-1).nonzero().squeeze(1)

        # num_subg_nodes
        subg_mask = node_mask.any(dim=1)
        flatten_subg_idxs = subg_mask.nonzero().squeeze(1)

        # num_subg_nodes * batch_size
        subg_node_mask = node_mask.index_select(0, flatten_subg_idxs)
        flatten_subg_node_idxs = subg_node_mask.view(-1).nonzero().squeeze(1)

        g = g.to(self.device)
        subg = g.subgraph(flatten_subg_idxs)

        # num_subg_nodes * batch_size * feature_dim
        h = self._build_h(ob, g).index_select(0, flatten_subg_idxs)
        #print('here')
        return (
            (node_mask, subg_mask, subg_node_mask),
            (flatten_node_idxs, flatten_subg_idxs, flatten_subg_node_idxs),
            subg,
            h
        )

    def act(self, ob, g):

        num_nodes, batch_size = ob.size(0), ob.size(1)
        #        g = g.reverse(share_ndata=True, share_edata=True).to(self.device)#将边改为反向
        #        g = dgl.DGLGraph(nx.DiGraph(nx.Graph(g.to_networkx())))  # 将边改为双向
        masks, idxs, subg, h = self.get_masks_idxs_subg_h(ob, g)
        node_mask, subg_mask, subg_node_mask = masks
        flatten_node_idxs, flatten_subg_idxs, flatten_subg_node_idxs = idxs

        # compute logits to get action
        logits = (
            self.actor_net(
                h,
                subg,
                mask=subg_node_mask
            )
                .view(-1, self.output_dim)  # 改
                .index_select(0, flatten_subg_node_idxs)
        )

        # get actions
        action = torch.zeros(
            num_nodes * batch_size,
            dtype=torch.long,
            device=self.device
        )
        m = Categorical(
            logits=logits.view(-1, logits.size(-1))
        )
        action[flatten_node_idxs] = m.sample()
        action = action.view(-1, batch_size)

        return action

    def act_and_crit(self, ob, g):
        num_nodes, batch_size = ob.size(0), ob.size(1)
        #        g = g.reverse(share_ndata=True, share_edata=True).to(self.device)#将边改为反向
        #        g = dgl.DGLGraph(nx.DiGraph(nx.Graph(g.to_networkx())))  # 将边改为双向
        masks, idxs, subg, h = self.get_masks_idxs_subg_h(ob, g)
        node_mask, subg_mask, subg_node_mask = masks
        flatten_node_idxs, flatten_subg_idxs, flatten_subg_node_idxs = idxs

        # compute logits to get action
        logits = (
            self.actor_net(
                h,
                subg,
                mask=subg_node_mask
            )
                .view(-1, self.output_dim)  # 改
                .index_select(0, flatten_subg_node_idxs)
        )
        #print(logits)

        m = Categorical(logits=logits)
        # get actions
        action = torch.zeros(
            num_nodes * batch_size,
            dtype=torch.long,
            device=self.device
        )
        action[flatten_node_idxs] = m.sample()  # 确定(sample) action

        # compute log probability of actions per node
        action_log_probs = torch.zeros(
            num_nodes * batch_size,
            device=self.device
        )
        action_log_probs[flatten_node_idxs] = m.log_prob(
            action.index_select(0, flatten_node_idxs)
        )

        action = action.view(-1, batch_size)
        action_log_probs = action_log_probs.view(-1, batch_size)

        # compute value predicted by critic
        node_value_preds = torch.zeros(
            num_nodes * batch_size,
            device=self.device
        )

        node_value_preds[flatten_node_idxs] = (
            self.critic_net(
                h,
                subg,
                mask=subg_node_mask
            )
                .view(-1)
                .index_select(0, flatten_subg_node_idxs)
        )

        g = g.to(self.device)
        g.ndata['h'] = node_value_preds.view(-1, batch_size)
        value_pred = dgl.sum_nodes(g, 'h') / self.max_num_nodes
        g.ndata.pop('h')

        return action, action_log_probs, value_pred

    def evaluate_batch(self, ob, g, action):
        num_nodes, batch_size = ob.size(0), ob.size(1)
        masks, idxs, subg, h = self.get_masks_idxs_subg_h(ob, g)
        node_mask, subg_mask, subg_node_mask = masks
        flatten_node_idxs, flatten_subg_idxs, flatten_subg_node_idxs = idxs

        # compute logits to get action
        logits = (
            self.actor_net(
                h,
                subg,
                mask=subg_node_mask
            )
                .view(-1, self.output_dim)
                .index_select(0, flatten_subg_node_idxs)
        )

        #print('ob',(ob.select(2, 0).long() == 0).float().max())


        m = Categorical(logits=logits)

        # compute log probability of actions per node
        action_log_probs = torch.zeros(
            num_nodes * batch_size,
            device=self.device
        )

        action_log_probs[flatten_node_idxs] = m.log_prob(
            action.reshape(-1).index_select(0, flatten_node_idxs)
        )
        action_log_probs = action_log_probs.view(-1, batch_size)

        node_entropies = - torch.sum(
            torch.softmax(logits, dim=1)
            * torch.log_softmax(logits, dim=1),
            dim=1
        )
        avg_entropy = node_entropies.mean()

        # compute value predicted by critic
        node_value_preds = torch.zeros(
            num_nodes * batch_size,
            device=self.device
        )

        node_value_preds[flatten_node_idxs] = (
            self.critic_net(
                h,
                subg,
                mask=subg_node_mask
            )
                .view(-1)
                .index_select(0, flatten_subg_node_idxs)
        )

        g = g.to(self.device)
        g.ndata['h'] = node_value_preds.view(num_nodes, batch_size)
        value_preds = dgl.sum_nodes(g, 'h') / self.max_num_nodes
        g.ndata.pop('h')

        return action_log_probs, avg_entropy, value_preds, node_mask

    def _build_h(self, ob, g):
        ob_t = ob.select(2, 1).unsqueeze(2)  # 第2维取1列 (5,10,1)
        ob_x = ob.select(2, 0)
        onehot = nn.functional.one_hot(ob_x.to(torch.int64), num_classes=self.num_color + 1).to(torch.float32)
        g.ndata['h'] = onehot[:, :, 1:]  # 将0的one hot删除
        g.update_all(
            fn.copy_src(src='h', out='m'),
            fn.sum(msg='m', out='h')
        )
        # 通过消息传递求degree
        neighbor_color = g.ndata.pop('h')  # 将ndata存进x1_deg并删除'h'
        # 反向传播
        rg = g.reverse(copy_ndata=True, copy_edata=True).to(self.device)
        rg.ndata['h'] = onehot[:, :, 1:]  # 将0的one hot删除
        rg.update_all(
            fn.copy_src(src='h', out='m'),
            fn.sum(msg='m', out='h')
        )
        # 通过消息传递求degree
        neighbor_color_reverse = rg.ndata.pop('h')  # 将ndata存进x1_deg并删除'h'

        return torch.cat([ob_t, neighbor_color, neighbor_color_reverse, torch.ones_like(ob_t)], dim=2)