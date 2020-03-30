from networkit import *
from scipy.linalg import block_diag
from scipy.sparse import csr_matrix
from scipy.stats import kendalltau
import scipy.sparse as sp
import networkx as nx
import numpy as np
import pickle
import torch

def get_out_edges(g_nkit, node_sequence):
  global all_out_dict
  all_out_dict = dict()
  for all_n in node_sequence:
    all_out_dict[all_n] = set()
  for all_n in node_sequence:
    _ = g_nkit.forEdgesOf(all_n, nkit_outedges)
            
  return all_out_dict

def get_in_edges(g_nkit, node_sequence):
  global all_in_dict
  all_in_dict = dict()
  for all_n in node_sequence:
    all_in_dict[all_n] = set()
  for all_n in node_sequence:
    _ = g_nkit.forInEdgesOf(all_n, nkit_inedges)
  return all_in_dict

def nkit_inedges(u, v, weight, edgeid):
  all_in_dict[u].add(v)

def nkit_outedges(u, v, weight, edgeid):
  all_out_dict[u].add(v)

def nx2nkit(g_nx):
  node_num = g_nx.number_of_nodes()
  g_nkit = Graph(directed=True)
    
  for i in range(node_num):
    g_nkit.addNode()
    
  for e1,e2 in g_nx.edges():
    g_nkit.addEdge(e1, e2)

  assert g_nx.number_of_nodes() == g_nkit.numberOfNodes(), 'Number of nodes not matching'
  assert g_nx.number_of_edges() == g_nkit.numberOfEdges(), 'Number of edges not matching'
        
  return g_nkit

def clique_check(index, node_sequence, all_out_dict, all_in_dict):
  node = node_sequence[index]
  in_nodes = all_in_dict[node]
  out_nodes = all_out_dict[node]

  for in_n in in_nodes:
    tmp_out_nodes = set(out_nodes)
    tmp_out_nodes.discard(in_n)
    if tmp_out_nodes.issubset(all_out_dict[in_n]) == False:
      return False
  return True

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
  """Convert a scipy sparse matrix to a torch sparse tensor."""
  sparse_mx = sparse_mx.tocoo().astype(np.float32)
  indices = torch.from_numpy(
    np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
  values = torch.from_numpy(sparse_mx.data)
  shape = torch.Size(sparse_mx.shape)
  return torch.sparse.FloatTensor(indices, values, shape)

def graph_to_adjacent_list(list_graph, list_n_sequence, list_node_num, model_size):
  list_adjacency = []
  list_adjacency_t = []
  list_degree = []
  zero_list = []
  list_sparse_diag = []
  max_nodes = model_size
    
  for i in range(len(list_graph)):
    graph = list_graph[i]
    edges = list(graph.edges())
    graph = nx.MultiDiGraph()
    graph.add_edges_from(edges)

    # remove self loop edges 
    self_loops = [i for i in nx.selfloop_edges(graph)]
    graph.remove_edges_from(self_loops)
    node_sequence = list_n_sequence[i]

    adjacent_temp = nx.adjacency_matrix(graph, nodelist=node_sequence)
    node_num = list_node_num[i]
    # convert transposed adjacent matrix
    adjacent_temp_t = adjacent_temp.transpose()
        
    arr_temp1 = np.sum(adjacent_temp,axis=1)
    arr_temp2 = np.sum(adjacent_temp_t,axis=1)
    arr_multi = np.multiply(arr_temp1,arr_temp2)
    arr_multi = np.where(arr_multi > 0, 1.0, 0.0)
    # compute the degree of every node
    degree_arr = arr_multi
        
    non_zero_ind = np.nonzero(degree_arr.flatten())
    non_zero_ind = non_zero_ind[0]
        
    g_nkit = nx2nkit(graph)

    in_n_seq = [node_sequence[nz_ind] for nz_ind in non_zero_ind]
    all_out_dict = get_out_edges(g_nkit, node_sequence)
    all_in_dict = get_in_edges(g_nkit, in_n_seq)
    # print(all_in_dict)
    # print(all_out_dict)

    # check clique and set the degree as 0
    for index in non_zero_ind:
      is_zero = clique_check(index, node_sequence, all_out_dict, all_in_dict)
      if is_zero == True:
        degree_arr[index, 0] = 0.0
                    
    adjacent_temp = adjacent_temp.multiply(csr_matrix(degree_arr))
    adjacent_temp_t = adjacent_temp_t.multiply(csr_matrix(degree_arr))
                

    top_mat = csr_matrix((0, 0))
    remain_ind = max_nodes - node_num
    bottom_mat = csr_matrix((remain_ind, remain_ind))
        
    # adding extra padding to adjacent matrix, normalize and save as torch tensor
    adjacent_temp = csr_matrix(adjacent_temp)
    adjacent_mat = sp.block_diag((top_mat, adjacent_temp, bottom_mat))
        
    adjacent_temp_t = csr_matrix(adjacent_temp_t)
    adjacent_mat_t = sp.block_diag((top_mat, adjacent_temp_t, bottom_mat))
        
    adjacent_mat = sparse_mx_to_torch_sparse_tensor(adjacent_mat)
    list_adjacency.append(adjacent_mat)
    adjacent_mat_t = sparse_mx_to_torch_sparse_tensor(adjacent_mat_t)
    list_adjacency_t.append(adjacent_mat_t)
  return list_adjacency, list_adjacency_t


def ranking_correlation(output, true_value, node_num, model_size, k):
  # reshape the incoming data
  output = output.reshape((model_size))
  true_value = true_value.reshape((model_size))

  # get the prediction
  predictions = output.cpu().detach().numpy()
  true_array = true_value.cpu().detach().numpy()

  top_k_predictions = predictions.argsort()[::-1][:k]
  top_k_true = true_array.argsort()[::-1][:k]
  accuracy = 0
  print(f'top_k_predictions: {top_k_predictions}')
  print(f'top_k_true: {top_k_true}')
  for item in top_k_predictions:
    if item in top_k_true:
      accuracy += 1


  kt,_ = kendalltau(predictions[:node_num], true_array[:node_num])

  return kt, (accuracy / k) * 100


def loss_calculate(output, true_value, num_nodes, device, model_size):
  # reshape the incoming data
  output = output.reshape((model_size))
  true_value = true_value.reshape((model_size))
  
  _, order_y_true = torch.sort(-true_value[:num_nodes])

  # sample the data
  sample_num = num_nodes * 20
  ind_1 = torch.randint(0, num_nodes, (sample_num, )).long().to(device)
  ind_2 = torch.randint(0, num_nodes, (sample_num, )).long().to(device)
    
  rank_measure = torch.sign(-1 * (ind_1 - ind_2)).float()
        
  input_arr1 = output[:num_nodes][order_y_true[ind_1]].to(device)
  input_arr2 = output[:num_nodes][order_y_true[ind_2]].to(device)

  # compute the loss by Margin Ranking Loss function
  loss_rank = torch.nn.MarginRankingLoss(margin=1.0).forward(input_arr1, input_arr2, rank_measure)
 
  return loss_rank
