import tensorflow as tf
import networkx as nx
import graph
import utils

EMBEDDING_SIZE = 128
LEARNING_RATE = 0.0001
BATCH_SIZE = 16

REG_HIDDEN = (int)(EMBEDDING_SIZE / 2)
MAX_PROPOGATION_ITERATION = 5

NUM_MIN = 100
NUM_MAX = 200
MAX_ITERATION = 10000

NUM_VALID = 100

class DrBC:
  def __init__(self):
    self.graph_type = 'powerlaw'
    self.utils = utils.py_Utils()
    self.embedding_size = EMBEDDING_SIZE
    self.learning_rate = LEARNING_RATE
    self.reg_hidden = REG_HIDDEN
    self.activation_function = tf.nn.leaky_relu


    self.training_graphs = []
    self.testing_graphs = []
    self.training_betweenness_list = []
    self.testing_betweenness_list = []

    self.node_feature = tf.placeholder(tf.float32, name='node_feature')
    self.aux_feature = tf.placeholder(tf.float32, name='aux_feature')
    self.n2nsum_param = tf.sparse_placeholder(tf.float64, name='n2nsum_param')
    self.label = tf.placeholder(tf.float32, shape=[None, 1], name='label')
    
    self.ranking_loss_source = tf.placeholder(tf.int32, shape=[1, None], name='ranking_loss_source')
    self.ranking_loss_target = tf.placeholder(tf.int32, shape=[1, None], name='ranking_loss_target')
    # build the training network
    self.loss, self.train_step, self.preditions, self.node_embedding, self.param_list = self.build_training_network()

    self.saver = tf.train.Saver(max_to_keep=None)
    config = tf.ConfigProto(device_count={'CPU': 8},
                            inter_op_parallelism_threads=100,
                            intra_op_parallelism_threads=100,
                            log_device_placement=False)
    config.gpu_options.allow_growth = True
    self.session = tf.Session(config=config)
    self.session.run(tf.global_variables_initializer())

  def Train(self):
    print('Generating validation graphs ...\n')
    self.clear_graph(is_testing=True)
    for i in range(NUM_VALID):
      self.generate_graphs(NUM_MIN, NUM_MAX, is_testing=True)
    for i in range(1000):
      self.generate_graphs(NUM_MIN, NUM_MAX, is_testing=False)

    save_dir = './models'
    validate_file = '%s/ValidValue.csv' % save_dir
    
    with open(validate_file, 'w') as fp:
      for iter in range(MAX_ITERATION):
        train_loss = self.Fit()
        # regenerate different graph for training after every 5000 steps
        if iter and iter % 5000 == 0:
          self.generate_graphs(NUM_MIN, NUM_MAX, is_testing=False)
        
        # evaluate current training result after every 500 steps
        if iter % 500 == 0:
          if iter == 0:
            N_start = time.clock()
          else:
            N_start = N_end
          
          # start to validate the current training result
          frac_topk, frac_kendal = 0.0, 0.0
          test_start = time.time()
          for index in range(NUM_VALID):
            run_time, temp_topk, temp_kendal = self.Test(index)
            frac_topk += temp_topk / NUM_VALID
            frac_kendal += temp_kendal / NUM_VALID
          test_end = time.time()
          # write the evaluate result to output file
          fp.write('%.6f, %.6f\n' % (frac_topk, frac_kendal))
          fp.flush()
          print('\niter %d, Top0.01: %.6f, kendal: %.6f'%(iter, frac_topk, frac_kendal))
          print('testing %d graphs time: %.2fs' % (NUM_VALID, test_end - test_start))
          N_end = time.clock()
          print('500 iterations total time: %.2fs' % (N_end - N_start))
          print('Training loss is %.4f' % train_loss)
          # save the model after every 500 steps
          model_path = '%s/nrange_iter_%d_%d_%d.ckpt' % (save_dir, NUM_MIN, NUM_MAX, iter)
          self.SaveModel(model_path)
    
  def Fit(self):
    pass
  def Test(self):
    pass
  def Predict(self):
    pass

  """
  The function to build the network for training and predicting
  """
  def build_training_network(self):
    weight_n2l = tf.Variable(tf.truncated_normal([3, self.embedding_size], stddev=0.01), tf.float32, name='weight_n2l')
    node_convolution = tf.Variable(tf.truncated_normal([self.embedding_size, self.embedding_size], stddev=0.01), tf.float32, name='node_convolution')

    # combine self embedding and neighbor embedding in GRU method
    w_r = tf.Variable(tf.truncated_normal([self.embedding_size, self.embedding_size], stddev=0.01), tf.float32,name='w_r')
    u_r = tf.Variable(tf.truncated_normal([self.embedding_size, self.embedding_size], stddev=0.01), tf.float32,name='u_r')
    w_z = tf.Variable(tf.truncated_normal([self.embedding_size, self.embedding_size], stddev=0.01), tf.float32,name='w_z')
    u_z = tf.Variable(tf.truncated_normal([self.embedding_size, self.embedding_size], stddev=0.01), tf.float32,name='u_z')
    w = tf.Variable(tf.truncated_normal([self.embedding_size, self.embedding_size], stddev=0.01), tf.float32,name='w')
    u = tf.Variable(tf.truncated_normal([self.embedding_size, self.embedding_size], stddev=0.01), tf.float32,name='u')

    weight_h1 = tf.Variable(tf.truncated_normal([self.embedding_size, self.reg_hidden], stddev=0.01), tf.float32, name='weight_h1')
    weight_h2 = tf.Variable(tf.truncated_normal([self.reg_hidden + 4, 1], stddev=0.01), tf.float32, name='weight_h2')

    node_size = tf.shape(self.n2nsum_param)[0]

    input_message = tf.matmul(tf.cast(self.node_feature, tf.float32), weight_n2l)

    level = 0
    current_message_layer = self.activation_function(input_message)
    current_message_layer = tf.nn.l2_normalize(current_message_layer, axis=1)
    current_message_layer_JK = current_message_layer

    while(level < MAX_PROPOGATION_ITERATION):
      level = level + 1
      pool_n2n = tf.sparse_tensor_dense_matmul(tf.cast(self.n2nsum_param, tf.float64), tf.cast(current_message_layer, tf.float64))
      pool_n2n = tf.cast(pool_n2n, tf.float32)

      node_linear = tf.matmul(pool_n2n, node_convolution)

      r_t = tf.nn.relu(tf.add(tf.matmul(node_linear, w_r), tf.matmul(current_message_layer, u_r)))
      z_t = tf.nn.relu(tf.add(tf.matmul(node_linear, w_z), tf.matmul(current_message_layer, u_z)))
      h_t = tf.nn.tanh(tf.add(tf.matmul(node_linear, w), tf.matmul(r_t * current_message_layer, u)))
      current_message_layer = (1 - z_t) * current_message_layer + z_t * h_t
      current_message_layer = tf.nn.l2_normalize(current_message_layer, axis=1)

      current_message_layer_JK = tf.maximum(current_message_layer_JK, current_message_layer)
      current_message_layer = tf.nn.l2_normalize(current_message_layer, axis=1)

    current_message_layer = current_message_layer_JK
    current_message_layer = tf.nn.l2_normalize(current_message_layer, axis=1)
    node_embedding = current_message_layer
    hidden = tf.matmul(node_embedding, weight_h1)
    last_output = self.activation_function(hidden)
    last_output = tf.concat([last_output, 4], axis=1)
    temp_predictions = tf.matmul(last_output, weight_h2)

    labels = tf.nn.embedding_lookup(self.label, self.ranking_loss_source) - tf.nn.embedding_lookup(self.label, self.ranking_loss_target)
    predictions = tf.nn.embedding_lookup(temp_predictions, self.ranking_loss_source) - tf.nn.embedding_lookup(temp_predictions, self.ranking_loss_target)

    loss = self.pairwise_ranking_loss(predictions, labels)
    train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(loss)

    return loss, train_step, temp_predictions, node_embedding, tf.trainable_variables()

  """
  The function to compute the pairwise ranking loss for evaluation on trained model
  """
  def pairwise_ranking_loss(self, predictions, labels):
    loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=preditions,
                                                    labels=tf.sigmoid(labels))
    loss = tf.reduce_sum(loss, axis=1)
    return tf.reduce_mean(loss)

  """
  The function to clear the graph network data
  """
  def clear_graph(self, is_testing):
    # clear diffferent graph network data based on the is_testing flag
    if not is_testing:
      self.training_graphs = []
      self.training_betweenness_list = []
    else:
      self.testing_graphs = []
      self.testing_betweenness_list = []
 
  """
  the function to generate the graph network for training
  """
  def generate_graphs(self, num_min, num_max, is_testing):
    # self.clear_graph(is_testing)
    # generate different graph network data based on the is_testing flag
    if not is_testing:
      for i in range(1000):
        current_number = np.random.randint(num_max - num_min + 1) + num_min
        temp_graph = nx.powerlaw_cluster_graph(n=current_number, m=4, p=0.05)
        self.training_graphs.append(graph)
        bc = self.utils.Betweenness(self.generate_graph_network(temp_graph))
        bc_log = self.utils.bc_log
        self.training_betweenness_list.append(bc_log)
    else:
      for i in range(1000):
        current_number = np.random.randint(num_max - num_min + 1) + num_min
        temp_graph = nx.powerlaw_cluster_graph(n=current_number, m=4, p=0.05)
        self.testing_graphs.append(temp_graph)
        bc = self.utils.Betweenness(self.generate_graph_network(temp_graph))
        bc_log = self.utils.bc_log
        self.testing_betweenness_list.append(bc_log)
  
  def generate_graph_network(self, target_graph):
    edges = target_graph.edges()
    if len(edges) > 0:
        a, b = zip(*edges)
        A = np.array(a)
        B = np.array(b)
    else:
        A = np.array([0])
        B = np.array([0])
    return graph.py_Graph(len(target_graph.nodes()), len(edges), A, B)