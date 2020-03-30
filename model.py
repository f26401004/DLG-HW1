from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn.functional as functional
import torch.nn as nn
import torch
import math


"""
The first GNN layer using in whole GNN model
"""
class GNN_Layer_Init(Module):
  def __init__(self, in_features, out_features, bias=True):
    super(GNN_Layer_Init, self).__init__()
    # variable configuration
    self.in_features = in_features
    self.out_features = out_features
    # initialize the weight as PyTorch tensor
    self.weight = Parameter(torch.FloatTensor(in_features, out_features))

    if bias:
      self.bias = Parameter(torch.FloatTensor(out_features))
    else:
      self.register_parameter('bias', None)
    self.reset_parameters()
    
  def __repr__(self):
    return self.__class__.__name__ + ' (' \
      + str(self.in_features) + ' -> ' \
      + str(self.out_features) + ')'


  def reset_parameters(self):
    # normalize the weight data
    stdv = 1. / math.sqrt(self.weight.size(1))
    self.weight.data.uniform_(-stdv, stdv)
    if self.bias is not None:
      self.bias.data.uniform_(-stdv, stdv)

  """The main logic to forward the current features to next layer"""
  def forward(self, adjacent):
    # output sparse matrix multiplication direclty
    output = torch.spmm(adjacent, self.weight)

    if self.bias is not None:
      return output + self.bias
    else:
      return output

  def __repr__(self):
    return self.__class__.__name__ + ' (' \
      + str(self.in_features) + ' -> ' \
      + str(self.out_features) + ')'


"""
The GNN layer using in whole GNN model except the first layer
"""
class GNN_Layer(Module):
  def __init__(self, in_features, out_features, bias=True):
    super(GNN_Layer, self).__init__()
    # variable configuration
    self.in_features = in_features
    self.out_features = out_features
    # initialize the weight as PyTorch tensor
    self.weight = Parameter(torch.FloatTensor(in_features, out_features))
    if bias:
      self.bias = Parameter(torch.FloatTensor(out_features))
    else:
      self.register_parameter('bias', None)
    self.reset_parameters()
    
  def reset_parameters(self):
    # normalize the weight data
    stdv = 1. / math.sqrt(self.weight.size(1))
    self.weight.data.uniform_(-stdv, stdv)
    if self.bias is not None:
      self.bias.data.uniform_(-stdv, stdv)

  """The main logic to forward the current features to next layer"""
  def forward(self, input, adjacent):
    # calculate the matrix multiplcation on input and weight matrix
    support = torch.mm(input, self.weight)
    # perform sparse matrix multiplication on support and adjacent matrix
    output = torch.spmm(adjacent, support)

    if self.bias is not None:
      return output + self.bias
    else:
      return output


class GNN_Bet(nn.Module):
  def __init__(self, node_input, node_hidden, dropout):
    super(GNN_Bet, self).__init__()

    # config 4 GNN layer for aggregating node features 
    self.graph_layer_1 = GNN_Layer_Init(node_input, node_hidden)
    self.graph_layer_2 = GNN_Layer(node_hidden, node_hidden)
    self.graph_layer_3 = GNN_Layer(node_hidden, node_hidden)
    self.graph_layer_4 = GNN_Layer(node_hidden, node_hidden)

    self.dropout = dropout

    # a linear multilayer perceptron to generate final score
    self.linear_1 = nn.Linear(node_hidden, node_hidden * 2)
    self.linear_2 = nn.Linear(node_hidden * 2, node_hidden * 2)
    self.linear_3 = nn.Linear(node_hidden * 2, 1)
  
  """The main logic to forward the current features to next layer"""
  def forward(self, in_adjacent_list, out_adjacent_list):
    # aggregate the in-degree node features
    x_in_1 = functional.normalize(functional.relu(self.graph_layer_1(in_adjacent_list)), p=2, dim=1)
    x_in_2 = functional.normalize(functional.relu(self.graph_layer_2(x_in_1, in_adjacent_list)), p=2, dim=1)
    x_in_3 = functional.normalize(functional.relu(self.graph_layer_3(x_in_2, in_adjacent_list)), p=2, dim=1)
    x_in_4 = functional.relu(self.graph_layer_4(x_in_3, in_adjacent_list))
    
    # aggregate the out-degree node features
    x_out_1 = functional.normalize(functional.relu(self.graph_layer_1(out_adjacent_list)), p=2, dim=1)
    x_out_2 = functional.normalize(functional.relu(self.graph_layer_2(x_out_1, out_adjacent_list)), p=2, dim=1)
    x_out_3 = functional.normalize(functional.relu(self.graph_layer_3(x_out_2, out_adjacent_list)), p=2, dim=1)
    x_out_4 = functional.relu(self.graph_layer_4(x_out_3, out_adjacent_list))


    # compute the score for x_1 graph network
    score_in_1 = functional.relu(self.linear_1(x_in_1))
    score_in_1 = functional.dropout(score_in_1, self.dropout)
    score_in_1 = functional.relu(self.linear_2(score_in_1))
    score_in_1 = functional.dropout(score_in_1, self.dropout)
    score_in_1 = self.linear_3(score_in_1)

    score_in_2 = functional.relu(self.linear_1(x_in_2))
    score_in_2 = functional.dropout(score_in_2, self.dropout)
    score_in_2 = functional.relu(self.linear_2(score_in_2))
    score_in_2 = functional.dropout(score_in_2, self.dropout)
    score_in_2 = self.linear_3(score_in_2)

    score_in_3 = functional.relu(self.linear_1(x_in_3))
    score_in_3 = functional.dropout(score_in_3, self.dropout)
    score_in_3 = functional.relu(self.linear_2(score_in_3))
    score_in_3 = functional.dropout(score_in_3, self.dropout)
    score_in_3 = self.linear_3(score_in_3)

    score_in_4 = functional.relu(self.linear_1(x_in_4))
    score_in_4 = functional.dropout(score_in_4, self.dropout)
    score_in_4 = functional.relu(self.linear_2(score_in_4))
    score_in_4 = functional.dropout(score_in_4, self.dropout)
    score_in_4 = self.linear_3(score_in_4)

    
    score_out_1 = functional.relu(self.linear_1(x_out_1))
    score_out_1 = functional.dropout(score_out_1, self.dropout)
    score_out_1 = functional.relu(self.linear_2(score_out_1))
    score_out_1 = functional.dropout(score_out_1, self.dropout)
    score_out_1 = self.linear_3(score_out_1)

    score_out_2 = functional.relu(self.linear_1(x_out_2))
    score_out_2 = functional.dropout(score_out_2, self.dropout)
    score_out_2 = functional.relu(self.linear_2(score_out_2))
    score_out_2 = functional.dropout(score_out_2, self.dropout)
    score_out_2 = self.linear_3(score_out_2)

    score_out_3 = functional.relu(self.linear_1(x_out_3))
    score_out_3 = functional.dropout(score_out_3, self.dropout)
    score_out_3 = functional.relu(self.linear_2(score_out_3))
    score_out_3 = functional.dropout(score_out_3, self.dropout)
    score_out_3 = self.linear_3(score_out_3)

    score_out_4 = functional.relu(self.linear_1(x_out_4))
    score_out_4 = functional.dropout(score_out_4, self.dropout)
    score_out_4 = functional.relu(self.linear_2(score_out_4))
    score_out_4 = functional.dropout(score_out_4, self.dropout)
    score_out_4 = self.linear_3(score_out_4)

    score_in = score_in_1 + score_in_2 + score_in_3 + score_in_4
    score_out = score_out_1 + score_out_2 + score_out_3 + score_out_4
    
    # combine in-degree and out-degree node features by multiplication
    return torch.mul(score_in, score_out)