import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 

from sklearn import cluster, mixture
from sklearn.metrics import adjusted_rand_score
from skopt import gp_minimize

import networkx as nx

from pulser import Pulse, Sequence, Register
from pulser_simulation import QutipEmulator
from pulser.devices import Chadoq2, DigitalAnalogDevice
from pulser.waveforms import InterpolatedWaveform
from scipy.optimize import minimize

from utils.clustering_utils import evaluate_silhouettes, build_matrix, visualize_dataset, clusters_from_bitstring
from utils.quantum_utils import plot_distribution, evaluate_mapping, qaa

import os 
import sys
import shutil
import logging

if __name__ == '__main__':
  logging.basicConfig(stream=sys.stdout, level=logging.INFO)
  logger = logging.getLogger(__name__)

  # setup of the output folder
  current_folder = os.path.dirname(os.path.abspath(__file__))
  output_folder = os.path.join(current_folder, 'pulser')

  if os.path.exists(output_folder):
     shutil.rmtree(output_folder)
  
  os.makedirs(output_folder, exist_ok=True)
  
  logger.info('Output folder established')

  # Dataset source:  
  # P. Fänti and S. Sieranoja  
  # K-means properties on six clustering benchmark datasets  
  # Applied Intelligence, 48 (12), 4743-4759, December 2018.  
    
  # It can be downloaded from [this](https://cs.joensuu.fi/sipu/datasets/) webpage, under section Shape sets > [Aggregation](https://cs.joensuu.fi/sipu/datasets/Aggregation.txt). 
  data_folder = os.path.join(current_folder, 'data')
  dataset = pd.read_csv(os.path.join(data_folder, 'dataset.csv'))
  logger.info('Dataset loaded')

  # for brevity, define a variable containing the coordinates
  points = dataset[['x', 'y']]

  original_silhouettes = evaluate_silhouettes(points, dataset['label'])
  visualize_dataset(dataset['x'], dataset['y'], dataset['label'], original_silhouettes, 'Original dataset', path=os.path.join(output_folder, 'original_dataset.png'))

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
      # ("BIRCH", birch),
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
    
  logger.info('Clustering algorithms run')

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

  # save cluster information
  clusters.to_csv(os.path.join(output_folder, 'clusters.csv'))
  logger.info('Cluster information saved')

  # save clusterized points
  dataset.to_csv(os.path.join(output_folder, 'clusterized_dataset.csv'))
  logger.info('Clusterized information saved')

  ############ Saving clusters ############
  for index, clustering_algorithm in enumerate(clustering_algorithms):
    name, _ = clustering_algorithm 
    filename = os.path.join(output_folder, f'{name}.png')
    visualize_dataset(dataset['x'], dataset['y'], dataset[name], clusters.loc[clusters['algorithm'] == name], title=name, path=filename)

  logger.info('Cluster plots saved')

  ############ Building adjacency matrix ############
  adjacency_matrix = build_matrix(dataset, clusters)

  # saving matrix in Cinecubo required format
  cinecubo = []
  for i in range(adjacency_matrix.shape[0]):
    modified_row = adjacency_matrix[i, i:]
    cinecubo.append(modified_row)

  cinecubo = np.array(cinecubo, dtype=object)
  with open(os.path.join(output_folder, 'adjacency_matrix_cinecubo.txt'), 'w') as fp:
    for row in cinecubo:
      row_str = ' '.join(map(str, row))
      fp.write(row_str + '\n')
  
  logger.info('Adjacency matrix computed')

  ############ Saving matrix as image ############
  plt.figure()
  plt.imshow(adjacency_matrix, cmap='gray')
  plt.xticks(np.arange(len(adjacency_matrix[0])), np.arange(len(adjacency_matrix[0])))
  plt.yticks(np.arange(len(adjacency_matrix[0])), np.arange(len(adjacency_matrix[0])))
  plt.grid(color='grey')
  plt.savefig(os.path.join(output_folder, 'adjacency_matrix_fig.png'))

  # save adjacency matrix
  np.savetxt(os.path.join(output_folder, 'adjacency_matrix.csv'), adjacency_matrix, delimiter=',')

  logger.info('Adjacency matrix saved')

  ############ Building graph ############
  G = nx.from_numpy_array(adjacency_matrix)

  # removing self loops for better readability
  G.remove_edges_from(nx.selfloop_edges(G))

  logger.info('Adjacency graph computed')

  pos = nx.spring_layout(G, k=3, seed=1506)
   
  plt.figure()
  nx.draw(G, 
          pos=pos,
          with_labels=True, 
          node_color='orange', 
          node_size=500, 
          font_size=12, 
          font_weight='bold'
          )
  
  plt.savefig(os.path.join(output_folder, 'adjacency_graph.png'))

  logger.info('Adjacency graph saved')

  # QUBO matrix is the adjacency matrix
  Q = adjacency_matrix
  shape = (len(Q), 2)
  costs = []
  np.random.seed(0)
  x0 = np.random.random(shape).flatten()
  res = minimize(
      evaluate_mapping,
      x0,
      args=(Q, shape),
      method="CG",
      tol=1e-9,
  )

  logger.info('Register built')
  coords = np.reshape(res.x, (len(Q), 2))

  qubits = dict(enumerate(coords))

  reg = Register(qubits)
  reg.draw(
      blockade_radius=Chadoq2.rydberg_blockade_radius(1.0),
      draw_graph=True,
      draw_half_radius=True,
      fig_name=os.path.join(output_folder, 'register.png'),
      show=False
  ) 
  logger.info('Register plot saved')

  # define search space for parameters 
  space = [(0.1, np.median(Q[Q > 0].flatten())), (-10, -1), (1000, 5000)]

  result = gp_minimize(func=lambda params: qaa(params, reg, Q), dimensions=space, n_calls=10, random_state=1506)

  logger.info('Hyperparameter optimization finished')

  best_params = result.x 
  best_value = result.fun 

  # print("Optimal parameters:", best_params)
  # print("Maximum cost value found:", best_value)

  Omega, delta_0, T = best_params

  # rounding T to the closest multiple of 4 to avoid warnings
  T = round(T / 4) * 4
  delta_f = -delta_0

  # save best parameters obtained with Bayesian optimization
  pd.DataFrame([[*best_params, best_value]]).to_csv(os.path.join(output_folder, 'optimized_params.csv'),index=False, header=['omega', 'delta_0', 'T', 'min_cost'])
  logger.info('Bayesian optimization parameters saved')

  adiabatic_pulse = Pulse(
  InterpolatedWaveform(T, [1e-9, Omega, 1e-9]),
  InterpolatedWaveform(T, [delta_0, 0, delta_f]),
  0,
  )

  seq = Sequence(reg, DigitalAnalogDevice)
  seq.declare_channel('ising', 'rydberg_global')
  seq.add(adiabatic_pulse, 'ising')
  seq.draw(fig_name=os.path.join(output_folder, 'sequence.png'), show=False)
  logger.info('Sequence plot saved')

  simul = QutipEmulator.from_sequence(seq)
  results = simul.run()
  final = results.get_final_state()
  count_dict = results.sample_final_state()
  logger.info('Simulation run')


  # plotting readings distribution
  plot_distribution(count_dict, 20, path=os.path.join(output_folder, 'occurrences.png'))
  logger.info('Occurences distribution plot saved')

  # saving raw occurrences
  count_df = pd.DataFrame(list(count_dict.items()), columns=['bitstrings', 'occurrences']).sort_values(by='occurrences', ascending=False)
  count_df.to_csv(os.path.join(output_folder, 'raw_results.csv'), sep=',', index=False)

  # plotting of resulting clusterizations
  bitstrings = list(sorted(count_dict, key=lambda k: count_dict[k], reverse=True))[:3]

  for bitstring in bitstrings:
      selected_points, selected_clusters = clusters_from_bitstring(bitstring, dataset, clusters)
      if selected_points.shape[0] == dataset.shape[0]:
          rand_scores[bitstring] = adjusted_rand_score(dataset['label'], selected_points['label'])

      visualize_dataset(
          selected_points['x'], 
          selected_points['y'], 
          selected_points['label'], 
          clusters.loc[clusters['cluster_name'].isin(list(map(lambda cluster: f'{cluster[1]}_{cluster[0]}', selected_clusters)))], 
          'QAA',
          path=os.path.join(output_folder, f'{bitstring}.png')
        )
    
  logger.info('Obtained clusterization plots saved')
  
  # rand score 
  plt.figure()
  plt.bar(rand_scores.keys(), rand_scores.values())
  plt.title('Rand scores comparison')
  plt.xlabel('Algorithm')
  plt.ylabel('Rand score')
  plt.xticks(rotation=90)

  plt.savefig(os.path.join(output_folder, 'rand_scores.png'), bbox_inches='tight')

  logger.info('Rand scores comparison plot saved')
  logger.info('Script end')