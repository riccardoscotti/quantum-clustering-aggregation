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

from utils.clustering_utils import evaluate_silhouettes

# Dataset source:  
# P. Fänti and S. Sieranoja  
# K-means properties on six clustering benchmark datasets  
# Applied Intelligence, 48 (12), 4743-4759, December 2018.  
  
# It can be downloaded from [this](https://cs.joensuu.fi/sipu/datasets/) webpage, under section Shape sets > [Aggregation](https://cs.joensuu.fi/sipu/datasets/Aggregation.txt). 

dataset = pd.read_csv('./data/dataset.csv')

# for brevity, define a variable containing the coordinates
points = dataset[['x', 'y']]

original_silhouettes = evaluate_silhouettes(points, dataset['label'])
# visualize_dataset(dataset['x'], dataset['y'], dataset['label'], original_silhouettes, 'Original dataset')