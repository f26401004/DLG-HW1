from tqdm import tqdm
import numpy as np
import networkx as nx
import pickle
import random

SYNTHETIC_DATA_PATH = './data/Synthetic/5000'
YOUTUBE_DATA_PATH = './data/youtube/'

def preprocess_synthetic_data():
  # preprocessing training data
  synthetic_networks = {
    'graph_list': [],
    'node_sequence_list': [],
    'node_num_list': [],
    'bc_matrix': []
  }

  for i in range(20):
    synthetic_networks['node_sequence_list'].append([])
    print(f'Start preprcoessing {i}.txt file to network graph ...')
    graph = nx.Graph()
    with open(f'{SYNTHETIC_DATA_PATH}/{i}.txt', 'r') as fp:
      for line in tqdm(fp):
        nodes = line.split('\t')
        from_node = int(nodes[0])
        to_node = int(nodes[1])
        graph.add_edge(from_node, to_node)

        if from_node not in synthetic_networks['node_sequence_list'][i]:
          synthetic_networks['node_sequence_list'][i].append(from_node)

    for j in range(5000):
      if j not in synthetic_networks['node_sequence_list'][i]:
        synthetic_networks['node_sequence_list'][i].append(j)

    synthetic_networks['graph_list'].append(graph)
    synthetic_networks['node_num_list'].append(len(synthetic_networks['node_sequence_list'][i]))

    synthetic_networks['bc_matrix'].append(np.zeros(5000))
    with open(f'{SYNTHETIC_DATA_PATH}/{i}_score.txt', 'r') as fp:
      for line in tqdm(fp):
        data = line.split('\t')
        node = int(data[0])
        score = float(data[1])
        synthetic_networks['bc_matrix'][i][node] = score
  
  synthetic_networks['bc_matrix'] = np.array(synthetic_networks['bc_matrix'])
    
  with open(f'{SYNTHETIC_DATA_PATH}/training_data.pickle', 'wb') as fp:
    pickle.dump(synthetic_networks, fp, protocol=pickle.HIGHEST_PROTOCOL)

  # preprocessing testing data
  synthetic_networks = {
    'graph_list': [],
    'node_sequence_list': [],
    'node_num_list': [],
    'bc_matrix': []
  }

  for i in range(10):
    synthetic_networks['node_sequence_list'].append([])
    
    print(f'Start preprcoessing {i + 20}.txt file to network graph ...')
    graph = nx.Graph()
    with open(f'{SYNTHETIC_DATA_PATH}/{i + 20}.txt', 'r') as fp:
      for line in tqdm(fp):
        nodes = line.split('\t')
        from_node = int(nodes[0])
        to_node = int(nodes[1])
        graph.add_edge(from_node, to_node)

        if from_node not in synthetic_networks['node_sequence_list'][i]:
          synthetic_networks['node_sequence_list'][i].append(from_node)
    
    for j in range(5000):
      if j not in synthetic_networks['node_sequence_list'][i]:
        synthetic_networks['node_sequence_list'][i].append(j)
    



    synthetic_networks['graph_list'].append(graph)
    synthetic_networks['node_num_list'].append(len(synthetic_networks['node_sequence_list'][i]))

    synthetic_networks['bc_matrix'].append(np.zeros(5000))
    with open(f'{SYNTHETIC_DATA_PATH}/{i + 20}_score.txt', 'r') as fp:
      for line in tqdm(fp):
        data = line.split('\t')
        node = int(data[0])
        score = float(data[1])
        synthetic_networks['bc_matrix'][i][node] = score
  
  synthetic_networks['bc_matrix'] = np.array(synthetic_networks['bc_matrix'])
    
  with open(f'{SYNTHETIC_DATA_PATH}/testing_data.pickle', 'wb') as fp:
    pickle.dump(synthetic_networks, fp, protocol=pickle.HIGHEST_PROTOCOL)

  print('Finish preprocessing all network file!')

def preprocess_youtube_data():

  youtube_network = {
    'graph': None,
    'node_sequence_list': [0],
    'node_num': 0,
    'bc_matrix': {}
  }

  print(f'Start preprcoessing com-youtube.txt file to network graph ...')
  graph = nx.Graph()
  prev_node = 0 
  with open(f'{YOUTUBE_DATA_PATH}/com-youtube.txt', 'r') as fp:
    for line in tqdm(fp):
      nodes = line.split(' ')
      from_node = int(nodes[0])
      to_node = int(nodes[1])
      graph.add_edge(from_node, to_node)

      if from_node != prev_node:
        youtube_network['node_sequence_list'].append(from_node)
        prev_node = from_node

  youtube_network['graph'] = graph
  youtube_network['node_num'] = len(youtube_network['node_sequence_list'])

  youtube_network['bc_matrix'] = np.zeros(1134890)
  with open(f'{YOUTUBE_DATA_PATH}/com-youtube_score.txt', 'r') as fp:
    for line in tqdm(fp):
      data = line.split(':\t')
      node = int(data[0])
      score = float(data[1])
      youtube_network['bc_matrix'][node] = score
      

  with open(f'{YOUTUBE_DATA_PATH}/training_data.pickle', 'wb') as fp:
    pickle.dump(youtube_network, fp, protocol=pickle.HIGHEST_PROTOCOL)

  print('Finish preprocessing all network file!')


if __name__ == '__main__':
  preprocess_synthetic_data()
  # preprocess_youtube_data()