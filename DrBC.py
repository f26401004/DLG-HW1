import tensorflow as tf

class DrBC:
  def __init__(self):
    self.training_graphs = []
    self.testing_graphs = []

  def build_network(self):
    pass
  def train(self):
    pass
  def fit(self):
    pass

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
    else:
      self.testing_graphs = []
 
  """
  the function to generate the graph network for training
  """
  def generate_graphs(self, num_min, num_max, is_testing):
    self.clear_graph(is_testing)
    # generate different graph network data based on the is_testing flag
    if not is_testing:
      for i in range(1000):
        current_number = np.random.randint(num_max - num_min + 1) + num_min
        graph = nx.powerlaw_cluster_graph(n=current_number, m=4, p=0.05)
        self.training_graphs.append(graph)
    else:
      for i in range(1000):
        current_number = np.random.randint(num_max - num_min + 1) + num_min
        graph = nx.powerlaw_cluster_graph(n=current_number, m=4, p=0.05)
        self.testing_graphs.append(graph)
    