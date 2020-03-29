import numpy as np
import pickle
import networkx as nx
import torch
from utils import *
import random
import torch.nn as nn
from model import GNN_Bet
torch.manual_seed(20)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_PATH = './data/'
MODEL_SIZE = 10000
HIDDEN = 20

model = GNN_Bet(node_input=MODEL_SIZE, node_hidden=HIDDEN, dropout=0.6)
model.to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
NUM_EPOCH = 1000

def start():
  # load the training data
  print('Loading data ...')
  with open('%s/training_data.pickle' % DATA_PATH, 'rb') as fp:
    training_graph_list, training_seq_list, training_num_node_list, training_bc_matrix = pickle.load(fp)
  with open('%s/testing_data.pickle' % DATA_PATH, 'rb') as fp:
    testing_graph_list, testing_seq_list, testing_num_node_list, testing_bc_matrix = pickle.load(fp)

  print('Convert graphs to adjacent list format...')
  training_adjacent_list, training_test_adjacent_list = graph_to_adjacent_list(training_graph_list, training_seq_list, training_num_node_list, MODEL_SIZE)
  testing_adjacent_list, testing_test_adjacent_list = graph_to_adjacent_list(testing_graph_list, testing_seq_list, testing_num_node_list, MODEL_SIZE)

  for e in range(NUM_EPOCH):
    print(f'Epoch number: {e + 1}')
    train(training_adjacent_list, training_test_adjacent_list, training_num_node_list, training_bc_matrix)
  with torch.no_grad():
    test(testing_adjacent_list, testing_test_adjacent_list, testing_num_node_list, testing_bc_matrix)

def train(training_adjacent_list, training_test_adjacent_list, training_num_node_list, training_bc_matrix):
  model.train()
  training_loss = 0

  for i in range(len(training_adjacent_list)):
    adjacent = training_adjacent_list[i]
    node_number = training_num_node_list[i]
    adjacent_test = training_test_adjacent_list[i]
    adjacent = adjacent.to(DEVICE)
    adjacent_test = adjacent_test.to(DEVICE)

    optimizer.zero_grad()

    output = model(adjacent, adjacent_test)
    true_array = torch.from_numpy(training_bc_matrix[:, i]).float()
    true_value = true_array.to(DEVICE)

    ranking_loss = loss_calculate(output, true_value, node_number, DEVICE, MODEL_SIZE)
    training_loss = training_loss + float(ranking_loss)
    ranking_loss.backward()
    optimizer.step()

def test(testing_adjacent_list, testing_test_adjacent_list, testing_num_node_list, testing_bc_matrix):
  model.eval()
  loss = 0
  list_kt = list()
  for i in range(len(testing_adjacent_list)):
    adjacent = testing_adjacent_list[i]
    adjacent_test = testing_adjacent_list[i]
    adjacent = adjacent.to(DEVICE)
    adjacent_test = adjacent_test.to(DEVICE)
    node_number = testing_num_node_list[i]
        
    output = model(adjacent, adjacent_test)
    
        
    true_array = torch.from_numpy(testing_bc_matrix[:, i]).float()
    true_value = true_array.to(DEVICE)
    
    kt = ranking_correlation(output, true_value, node_number, MODEL_SIZE)
    list_kt.append(kt)

  print(f'Average kendall tau score is: {np.mean(np.array(list_kt))} and std: {np.std(np.array(list_kt))}')


if __name__ == '__main__':
  start()
