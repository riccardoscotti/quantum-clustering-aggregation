import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 

from sklearn import cluster, mixture
from sklearn.metrics import silhouette_samples, adjusted_rand_score
from skopt import gp_minimize

import networkx as nx

from itertools import islice, cycle

from pulser import Pulse, Sequence, Register
from pulser_simulation import QutipEmulator
from pulser.devices import Chadoq2, DigitalAnalogDevice
from pulser.waveforms import InterpolatedWaveform
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform

from utils.clustering_utils import evaluate_silhouettes, build_matrix

if __name__ == '__main__':
  # Dataset source:  
  # P. Fänti and S. Sieranoja  
  # K-means properties on six clustering benchmark datasets  
  # Applied Intelligence, 48 (12), 4743-4759, December 2018.  
    
  # It can be downloaded from [this](https://cs.joensuu.fi/sipu/datasets/) webpage, under section Shape sets > [Aggregation](https://cs.joensuu.fi/sipu/datasets/Aggregation.txt). 

  dataset = pd.read_csv('./data/dataset.csv')

  # for brevity, define a variable containing the coordinates
  points = dataset[['x', 'y']]

  original_silhouettes = evaluate_silhouettes(points, dataset['label'])
  visualize_dataset(dataset['x'], dataset['y'], dataset['label'], original_silhouettes, 'Original dataset')

  clustering_seed = 1506

  # DBSCAN
  eps = 1.2
  dbscan = cluster.DBSCAN(eps=eps)

  # Spectral Clustering
  spectral = cluster.SpectralClustering(n_clusters=5)

  # MeanShift
  bandwidth = cluster.estimate_bandwidth(points, quantile=0.12)
  ms = cluster.MeanShift(bandwidth=bandwidth, bin_seeding=True)

  # KMeans
  kmeans = cluster.KMeans(
      n_clusters=7,
      random_state=clustering_seed
  )

  # Birch
  birch = cluster.Birch(n_clusters=7, branching_factor=5)

  # NOTE: please uncomment the following lines one at a time
  clustering_algorithms = (
      ("DBSCAN", dbscan),
      ("Spectral Clustering", spectral),
      # ("MeanShift", ms),
      # ("KMeans", kmeans),
      # ("Birch", birch),
  )

  ############ Running algorithms ############
  labels_offset = 0
  for clustering_algorithm in clustering_algorithms:
    name, algorithm = clustering_algorithm

    algorithm.fit(points)
    
    if hasattr(algorithm, 'labels_'):
        dataset[name] = algorithm.labels_.astype(int)
        if algorithm.labels_.astype(int).__contains__(-1):
            dataset[name] += 1
    else:
        dataset[name] = algorithm.predict(points)
    
    dataset.loc[dataset[name] >= 0, name] += labels_offset
    labels_offset = dataset[name].max() + 1

  ############ Silhouette and rand score ############
  silhouette_df = []
  rand_scores = {}
  for clustering_algorithm in clustering_algorithms:
      name, _ = clustering_algorithm 

      rand_scores[name] = adjusted_rand_score(dataset['label'], dataset[name])

      silhouettes = pd.DataFrame()
      silhouettes = evaluate_silhouettes(points, dataset[name])
      silhouettes['algorithm'] = [name] * len(set(silhouettes['label'].values))
      silhouettes['cluster_name'] = silhouettes['algorithm'] + '_' + silhouettes['label'].astype(str)
      silhouette_df.append(silhouettes)

  clusters = pd.concat(silhouette_df, ignore_index=True)

  ############ Saving first three clusters ############
  for index, clustering_algorithm in enumerate(clustering_algorithms):
    name, _ = clustering_algorithm 
    visualize_dataset(dataset['x'], dataset['y'], dataset[name], clusters.loc[clusters['algorithm'] == name], title=name)

  ############ Building adjacency matrix ############
  adjacency_matrix = build_matrix(dataset, clusters)

  # saving matrix in Cinecubo required format
  cinecubo = []
  for i in range(adjacency_matrix.shape[0]):
    modified_row = adjacency_matrix[i, i:]
    cinecubo.append(modified_row)

  cinecubo = np.array(cinecubo, dtype=object)
  with open('./matrices/clustering_aggregation.txt', 'w') as fp:
    for row in cinecubo:
      row_str = ' '.join(map(str, row))
      fp.write(row_str + '\n')

  ############ Saving matrix as image ############
  plt.figure()
  plt.imshow(adjacency_matrix, cmap='gray')
  plt.xticks(np.arange(len(adjacency_matrix[0])), np.arange(len(adjacency_matrix[0])))
  plt.yticks(np.arange(len(adjacency_matrix[0])), np.arange(len(adjacency_matrix[0])))
  plt.grid(color='grey')
  plt.show()

  ############ Building of graph ############
  G = nx.from_numpy_array(adjacency_matrix)

  # removing self loops for better readability
  G.remove_edges_from(nx.selfloop_edges(G))

  pos = nx.spring_layout(G, k=3, seed=1506)

  nx.draw(G, 
          pos=pos,
          with_labels=True, 
          node_color='orange', 
          node_size=500, 
          font_size=12, 
          font_weight='bold'
          )