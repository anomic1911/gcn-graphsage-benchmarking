import torch
import torch.nn as nn
import torch_geometric as tg
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree, add_remaining_self_loops
from torch.nn import init
import pdb
from torch.nn import Parameter
import math
####################### Basic Ops ############################
def uniform(size, tensor):
    bound = 1.0 / math.sqrt(size)
    if tensor is not None:
        tensor.data.uniform_(-bound, bound)

class SAGEConvOpt(MessagePassing):
 
    def __init__(self, in_channels, out_channels, aggregator, normalize=False,
                 concat=False, bias=True, **kwargs):
        super(SAGEConvOpt, self).__init__(aggr=aggregator, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize
        self.concat = concat

        in_channels = 2 * in_channels if concat else in_channels
        self.weight = Parameter(torch.Tensor(in_channels, out_channels))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        uniform(self.weight.size(0), self.weight)
        uniform(self.weight.size(0), self.bias)

    def forward(self, x, edge_index, edge_weight=None, size=None,
                res_n_id=None):

        if not self.concat and torch.is_tensor(x):
            edge_index, edge_weight = add_remaining_self_loops(
                edge_index, edge_weight, 1, x.size(self.node_dim))

        return self.propagate(edge_index, size=size, x=x,
                              edge_weight=edge_weight, res_n_id=res_n_id)

    def message(self, x_j, edge_weight):
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def update(self, aggr_out, x, res_n_id):
        if self.concat and torch.is_tensor(x):
            aggr_out = torch.cat([x, aggr_out], dim=-1)
        elif self.concat and (isinstance(x, tuple) or isinstance(x, list)):
            assert res_n_id is not None
            aggr_out = torch.cat([x[0][res_n_id], aggr_out], dim=-1)

        aggr_out = torch.matmul(aggr_out, self.weight)

        if self.bias is not None:
            aggr_out = aggr_out + self.bias

        if self.normalize:
            aggr_out = F.normalize(aggr_out, p=2, dim=-1)

        return aggr_out


### Non linearity
class Nonlinear(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Nonlinear, self).__init__()

        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, output_dim)

        self.act = nn.ReLU()

        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data = init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    m.bias.data = init.constant_(m.bias.data, 0.0)

    def forward(self, x):
        x = self.linear1(x)
        x = self.act(x)
        x = self.linear2(x)
        return x




####################### NNs #############################

class MLP(torch.nn.Module):
    def __init__(self, input_dim, feature_dim, hidden_dim, output_dim,
                 feature_pre=True, layer_num=2, dropout=True, **kwargs):
        super(MLP, self).__init__()
        self.feature_pre = feature_pre
        self.layer_num = layer_num
        self.dropout = dropout
        if feature_pre:
            self.linear_pre = nn.Linear(input_dim, feature_dim)
            self.linear_first = nn.Linear(feature_dim, hidden_dim)
        else:
            self.linear_first = nn.Linear(input_dim, hidden_dim)
        self.linear_hidden = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for i in range(layer_num - 2)])
        self.linear_out = nn.Linear(hidden_dim, output_dim)


    def forward(self, data):
        x = data.x
        if self.feature_pre:
            x = self.linear_pre(x)
        x = self.linear_first(x)
        x = F.relu(x)
        if self.dropout:
            x = F.dropout(x, training=self.training)
        for i in range(self.layer_num - 2):
            x = self.linear_hidden[i](x)
            x = F.relu(x)
            if self.dropout:
                x = F.dropout(x, training=self.training)
        x = self.linear_out(x)
        x = F.normalize(x, p=2, dim=-1)
        return x


class GCN(torch.nn.Module):
    def __init__(self, input_dim, feature_dim, hidden_dim, output_dim,
                 feature_pre=True, layer_num=2, dropout=True, **kwargs):
        super(GCN, self).__init__()
        self.feature_pre = feature_pre
        self.layer_num = layer_num
        self.dropout = dropout
        if feature_pre:
            self.linear_pre = nn.Linear(input_dim, feature_dim)
            self.conv_first = tg.nn.GCNConv(feature_dim, hidden_dim)
        else:
            self.conv_first = tg.nn.GCNConv(input_dim, hidden_dim)
        self.conv_hidden = nn.ModuleList([tg.nn.GCNConv(hidden_dim, hidden_dim) for i in range(layer_num - 2)])
        self.conv_out = tg.nn.GCNConv(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        if self.feature_pre:
            x = self.linear_pre(x)
        x = self.conv_first(x, edge_index)
        x = F.relu(x)
        if self.dropout:
            x = F.dropout(x, training=self.training)
        for i in range(self.layer_num-2):
            x = self.conv_hidden[i](x, edge_index)
            x = F.relu(x)
            if self.dropout:
                x = F.dropout(x, training=self.training)
        x = self.conv_out(x, edge_index)
        x = F.normalize(x, p=2, dim=-1)
        return x

class SAGE(torch.nn.Module):
    def __init__(self, input_dim, feature_dim, hidden_dim, output_dim, aggregator = 'mean',
                 feature_pre=True, layer_num=2, dropout=True, **kwargs):
        super(SAGE, self).__init__()
        self.feature_pre = feature_pre
        self.layer_num = layer_num
        self.dropout = dropout
        if feature_pre:
            self.linear_pre = nn.Linear(input_dim, feature_dim)
            self.conv_first = SAGEConvOpt(feature_dim, hidden_dim, aggregator)
        else:
            self.conv_first = SAGEConvOpt(input_dim, hidden_dim, aggregator)
        self.conv_hidden = nn.ModuleList([SAGEConvOpt(hidden_dim, hidden_dim, aggregator) for i in range(layer_num - 2)])
        self.conv_out = SAGEConvOpt(hidden_dim, output_dim, aggregator)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        if self.feature_pre:
            x = self.linear_pre(x)
        x = self.conv_first(x, edge_index)
        x = F.relu(x)
        if self.dropout:
            x = F.dropout(x, training=self.training)
        for i in range(self.layer_num-2):
            x = self.conv_hidden[i](x, edge_index)
            x = F.relu(x)
            if self.dropout:
                x = F.dropout(x, training=self.training)
        x = self.conv_out(x, edge_index)
        x = F.normalize(x, p=2, dim=-1)
        return x

class PairwiseModel(torch.nn.Module):
    
    def __init__(self, input_dim, feature_dim, hidden_dim, output_dim, model_type = "GCN", 
                 aggregator = 'mean', feature_pre=True, layer_num=2, dropout=True, **kwargs):
        super(PairwiseModel, self).__init__()
        
        if model_type == "GCN":
            self.main_model = GCN(input_dim=input_dim, feature_dim=feature_dim, 
                             hidden_dim=hidden_dim, output_dim=output_dim)
            
        elif model_type == "GraphSAGE":
            self.main_model = SAGE(input_dim=input_dim, feature_dim=feature_dim, 
                             hidden_dim=hidden_dim, output_dim=output_dim, aggregator=aggregator)
            
        else:
            raise ValueError("Not a Valid Graph Neural Network")
            
    def forward(self, x):
        y = self.main_model(x)
        y = torch.matmul(y, y.T)
        y = F.sigmoid(y)
        return y


class MulticlassModel(torch.nn.Module):
    
    def __init__(self, input_dim, feature_dim, hidden_dim, output_dim, model_type = "GCN", 
                 aggregator = 'mean', feature_pre=True, layer_num=2, dropout=True, **kwargs):
        super(MulticlassModel, self).__init__()
        
        if model_type == "GCN":
            self.main_model = GCN(input_dim=input_dim, feature_dim=feature_dim, 
                             hidden_dim=hidden_dim, output_dim=output_dim)
            
        elif model_type == "GraphSAGE":
            self.main_model = SAGE(input_dim=input_dim, feature_dim=feature_dim, 
                             hidden_dim=hidden_dim, output_dim=output_dim, aggregator=aggregator)
            
        else:
            raise ValueError("Not a Valid Graph Neural Network")

        self.final = nn.Sigmoid()
            
    def forward(self, x):
        y = self.main_model(x)
        y = self.final(y)
        return y
        