from utils import *
from model import GNN_Bet
import numpy as np
import networkx as nx
import torch.nn as nn
import pickle
import torch
import random
import time
torch.manual_seed(20)

# config global variables
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_PATH = './data/Synthetic/5000'
# DATA_PATH = './data/'
MODEL_SIZE = 5000
HIDDEN = 16
NUM_EPOCH = 10000
TOP_K_NUMS = [1, 5, 10]

# initialize model in global scope
model = GNN_Bet(node_input=MODEL_SIZE, node_hidden=HIDDEN, dropout=0.6)
# attach model to out device
model.to(DEVICE)
# use Adam optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)


def start():
  # load the training data
  print('Loading data ...')

  with open('%s/training_data.pickle' % DATA_PATH, 'rb') as fp:
    data = pickle.load(fp)
    training_graph_list = data['graph_list']
    training_seq_list = data['node_sequence_list']
    training_num_node_list = data['node_num_list']
    training_bc_matrix = data['bc_matrix']

  with open('%s/testing_data.pickle' % DATA_PATH, 'rb') as fp:
    data = pickle.load(fp)
    testing_graph_list = data['graph_list']
    testing_seq_list = data['node_sequence_list']
    testing_num_node_list = data['node_num_list']
    testing_bc_matrix = data['bc_matrix']

  print('Convert graphs to adjacent list format...')
  training_adjacent_list, training_test_adjacent_list = graph_to_adjacent_list(training_graph_list, training_seq_list, training_num_node_list, MODEL_SIZE)
  testing_adjacent_list, testing_test_adjacent_list = graph_to_adjacent_list(testing_graph_list, testing_seq_list, testing_num_node_list, MODEL_SIZE)

  for i in range(NUM_EPOCH):
    print(f'Epoch number: {i + 1}')
    # shffle the graph data for each epoch
    temp = list(zip(training_adjacent_list, training_test_adjacent_list, training_bc_matrix))
    random.shuffle(temp)
    training_adjacent_list, training_test_adjacent_list, training_bc_matrix = zip(*temp)

    train(training_adjacent_list, training_test_adjacent_list, training_num_node_list, training_bc_matrix)
  
  # test top ks
  for k in TOP_K_NUMS:
    print(f'Testing for top {k}')
    # shffle the graph data for each epoch
    temp = list(zip(testing_adjacent_list, testing_test_adjacent_list, testing_bc_matrix))
    random.shuffle(temp)
    testing_adjacent_list, testing_test_adjacent_list, testing_bc_matrix = zip(*temp)

    # compute the testing duration
    test_start = time.time()
    with torch.no_grad():
      test(testing_adjacent_list, testing_test_adjacent_list, testing_num_node_list, testing_bc_matrix, k)
    test_duration = time.time() - test_start
    print(f'testing top-{k} graph times: {test_duration}')

def train(training_adjacent_list, training_test_adjacent_list, training_num_node_list, training_bc_matrix):
  # start training
  model.train()
  training_loss = 0

  for i in range(len(training_adjacent_list)):
    adjacent = training_adjacent_list[i]
    node_number = training_num_node_list[i]
    adjacent_test = training_test_adjacent_list[i]
    # attach adjacent matrix data to device
    adjacent = adjacent.to(DEVICE)
    adjacent_test = adjacent_test.to(DEVICE)

    optimizer.zero_grad()

    output = model(adjacent, adjacent_test)
    true_array = torch.from_numpy(training_bc_matrix[i]).float()
    true_value = true_array.to(DEVICE)

    # calculate the ranking loss and training loss
    ranking_loss = loss_calculate(output, true_value, node_number, DEVICE, MODEL_SIZE)
    training_loss = training_loss + float(ranking_loss)
    # backward propagation
    ranking_loss.backward()
    optimizer.step()
  
  print(f'training loss: {training_loss}')

def test(testing_adjacent_list, testing_test_adjacent_list, testing_num_node_list, testing_bc_matrix, top_k):
  # start evaluation
  model.eval()

  list_kt = []
  list_accuracy = []
  for i in range(len(testing_adjacent_list)):

    adjacent = testing_adjacent_list[i]
    adjacent_test = testing_adjacent_list[i]
    # attach adjacent matrix data to device
    adjacent = adjacent.to(DEVICE)
    adjacent_test = adjacent_test.to(DEVICE)
    node_number = testing_num_node_list[i]
        
    output = model(adjacent, adjacent_test)
    
        
    true_array = torch.from_numpy(testing_bc_matrix[i]).float()
    true_value = true_array.to(DEVICE)
    
    # calculate the Kendall tau distance and training loss
    kt, accuracy = ranking_correlation(output, true_value, node_number, MODEL_SIZE, top_k)
    list_kt.append(kt)
    list_accuracy.append(accuracy)

  print(f'Average kendall tau score is: {np.mean(np.array(list_kt))} and std: {np.std(np.array(list_kt))}')
  print(f'Average top-{top_k} accuracy: {np.mean(list_accuracy)}% and std: {np.std(list_accuracy)}')

if __name__ == '__main__':
  start()
