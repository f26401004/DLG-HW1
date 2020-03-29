import torch.nn as nn
import torch.nn.functional as functional
from layer import GNN_Layer
from layer import GNN_Layer_Init
import torch

class GNN_Bet(nn.Module):
  def __init__(self, node_input, node_hidden, dropout):
    super(GNN_Bet, self).__init__()

    self.graph_layer_1 = GNN_Layer_Init(node_input, node_hidden)
    self.graph_layer_2 = GNN_Layer(node_hidden, node_hidden)
    self.graph_layer_3 = GNN_Layer(node_hidden, node_hidden)
    self.graph_layer_4 = GNN_Layer(node_hidden, node_hidden)

    self.dropout = dropout

    self.linear_1 = nn.Linear(node_hidden, node_hidden * 2)
    self.linear_2 = nn.Linear(node_hidden * 2, node_hidden * 2)
    self.linear_3 = nn.Linear(node_hidden * 2, 1)
  
  def forward(self, adjacent_1, adjacent_2):
    x_1_1 = functional.normalize(functional.relu(self.graph_layer_1(adjacent_1)), p=2, dim=1)
    x_2_1 = functional.normalize(functional.relu(self.graph_layer_1(adjacent_2)), p=2, dim=1)

    x_1_2 = functional.normalize(functional.relu(self.graph_layer_2(x_1_1, adjacent_1)), p=2, dim=1)
    x_2_2 = functional.normalize(functional.relu(self.graph_layer_2(x_2_1, adjacent_2)), p=2, dim=1)

    x_1_3 = functional.normalize(functional.relu(self.graph_layer_3(x_1_2, adjacent_1)), p=2, dim=1)
    x_2_3 = functional.normalize(functional.relu(self.graph_layer_3(x_2_2, adjacent_2)), p=2, dim=1)

    x_1_4 = functional.relu(self.graph_layer_4(x_1_3, adjacent_1))
    x_2_4 = functional.relu(self.graph_layer_4(x_2_3, adjacent_2))

    # compute the score for x_1 graph network
    score_1_1 = functional.relu(self.linear_1(x_1_1))
    score_1_1 = functional.dropout(score_1_1, self.dropout)
    score_1_1 = functional.relu(self.linear_2(score_1_1))
    score_1_1 = functional.dropout(score_1_1, self.dropout)
    score_1_1 = self.linear_3(score_1_1)

    score_1_2 = functional.relu(self.linear_1(x_1_2))
    score_1_2 = functional.dropout(score_1_2, self.dropout)
    score_1_2 = functional.relu(self.linear_2(score_1_2))
    score_1_2 = functional.dropout(score_1_2, self.dropout)
    score_1_2 = self.linear_3(score_1_2)

    score_1_3 = functional.relu(self.linear_1(x_1_3))
    score_1_3 = functional.dropout(score_1_3, self.dropout)
    score_1_3 = functional.relu(self.linear_2(score_1_3))
    score_1_3 = functional.dropout(score_1_3, self.dropout)
    score_1_3 = self.linear_3(score_1_3)

    score_1_4 = functional.relu(self.linear_1(x_1_4))
    score_1_4 = functional.dropout(score_1_4, self.dropout)
    score_1_4 = functional.relu(self.linear_2(score_1_4))
    score_1_4 = functional.dropout(score_1_4, self.dropout)
    score_1_4 = self.linear_3(score_1_4)

    # compute the score for x_2 graph network
    score_2_1 = functional.relu(self.linear_1(x_2_1))
    score_2_1 = functional.dropout(score_2_1, self.dropout)
    score_2_1 = functional.relu(self.linear_2(score_2_1))
    score_2_1 = functional.dropout(score_2_1, self.dropout)
    score_2_1 = self.linear_3(score_2_1)

    score_2_2 = functional.relu(self.linear_1(x_2_2))
    score_2_2 = functional.dropout(score_2_2, self.dropout)
    score_2_2 = functional.relu(self.linear_2(score_2_2))
    score_2_2 = functional.dropout(score_2_2, self.dropout)
    score_2_2 = self.linear_3(score_2_2)

    score_2_3 = functional.relu(self.linear_1(x_2_3))
    score_2_3 = functional.dropout(score_2_3, self.dropout)
    score_2_3 = functional.relu(self.linear_2(score_2_3))
    score_2_3 = functional.dropout(score_2_3, self.dropout)
    score_2_3 = self.linear_3(score_2_3)

    score_2_4 = functional.relu(self.linear_1(x_2_4))
    score_2_4 = functional.dropout(score_2_4, self.dropout)
    score_2_4 = functional.relu(self.linear_2(score_2_4))
    score_2_4 = functional.dropout(score_2_4, self.dropout)
    score_2_4 = self.linear_3(score_2_4)

    score_1 = score_1_1 + score_1_2 + score_1_3 + score_1_4
    score_2 = score_2_1 + score_2_2 + score_2_3 + score_2_4

    return torch.mul(score_1, score_2)