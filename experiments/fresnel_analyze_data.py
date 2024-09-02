import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn import cluster
from sklearn.metrics import adjusted_rand_score
import json
import os 
import sys 
import logging

from utils.clustering_utils import clusters_from_bitstring, visualize_dataset
from utils.quantum_utils import plot_distribution

if __name__ == '__main__':
  logging.basicConfig(stream=sys.stdout, level=logging.INFO)
  logger = logging.getLogger(__name__)
  current_folder = os.path.dirname(os.path.abspath(__file__))
  output_folder = os.path.join(current_folder, 'fresnel')

  # importing of dataset (clusterized)
  dataset = pd.read_csv(os.path.join(output_folder, 'clusterized_dataset.csv'))

  # import clusters 
  clusters = pd.read_csv(os.path.join(output_folder, 'clusters.csv'))

  dbscan = cluster.DBSCAN(eps=1.2)

  # Spectral Clustering
  spectral = cluster.SpectralClustering(n_clusters=5)

  clustering_algorithms = (
      ("DBSCAN", dbscan),
      ("Spectral Clustering", spectral),
      # ("MeanShift", ms),
      # ("KMeans", kmeans),
      # ("BIRCH", birch),
  )

  raw = ''
  with open(os.path.join(output_folder, 'raw_results.json')) as fp:
    raw = json.load(fp).values()

  count_dict = next(iter(raw))['counter']
  plot_distribution(count_dict, 20, os.path.join(output_folder, 'occurrences.png'))

  logger.info('Occurences distribution plot saved')

  # rand scores 
  rand_scores = {}
  for clustering_algorithm in clustering_algorithms:
      name, _ = clustering_algorithm 

      rand_scores[name] = adjusted_rand_score(dataset['label'], dataset[name])

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