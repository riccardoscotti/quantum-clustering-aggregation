import numpy as np
from itertools import islice, cycle
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_samples
import pandas as pd

def get_colors(labels):
    colors = np.array(
        list(
            islice(
                cycle([
                    "#377eb8",
                    "#ff7f00",
                    "#4daf4a",
                    "#f781bf",
                    "#a65628",
                    "#984ea3",
                    "#999999",
                    "#e41a1c",
                    "#dede00",
                ]),
                int(max(labels) + 1),
            )
        )
    )
    
    # add black color for outliers (if any)
    colors = np.append(colors, ["#000000"])

    return colors[labels]

def visualize_dataset(x_points, y_points, labels, silhouettes, title, path=None):
    aspect_ratio = (max(x_points) - min(x_points)) / (max(y_points) - min(y_points))
    plt.figure(figsize=(8, 8 / aspect_ratio))
    plt.scatter(x_points, y_points, color=get_colors(labels))
    plt.title(f'{title} ({x_points.shape[0]} points, total sil. {silhouettes["silhouette"].sum():.2f})')
    plt.grid()

    silhouettes = silhouettes.set_index('label')['silhouette'].to_dict()
        
    cluster_legend = [
        plt.Line2D([0], 
                   [1], 
                   marker='o', 
                   color='w', 
                   markersize=10, 
                   markerfacecolor=get_colors([cluster])[0], 
                   label=f'Cluster {cluster} (sil. {silhouettes[cluster]:.2f})') 
        for cluster in set(labels)]
    

    plt.legend(handles=cluster_legend, title='Cluster', bbox_to_anchor=(1.05,1))

    if path:
        plt.savefig(path, bbox_inches='tight', format='pdf')
    else:
        plt.show()

def evaluate_silhouettes(points, labels):
    clusterization = pd.DataFrame({'label': labels})
    clusterization['silhouette'] = silhouette_samples(points, labels)
    
    silhouettes = clusterization.groupby('label')['silhouette'].sum().reset_index()

    return silhouettes

def get_points(points_df, algorithm, label):
    query = points_df.loc[points_df[algorithm] == label]
    return set(list(zip(query['x'], query['y'])))

def build_matrix(dataset, clusters):
    n = clusters['label'].nunique()
    adjacency_matrix = np.eye(n, dtype=int)
    silhouettes = clusters[['label', 'silhouette']]
    for i in range(n):
        for j in range(i + 1, n):
            i_algorithm = clusters.iloc[i]['algorithm']
            j_algorithm = clusters.iloc[j]['algorithm']

            i_silhouette = silhouettes.iloc[i]['silhouette']
            j_silhouette = silhouettes.iloc[j]['silhouette']


            # adjacency_matrix[i][j] = len(get_points(dataset, i_algorithm, i) & get_points(dataset, j_algorithm, j)) 
            adjacency_matrix[i][j] = len(get_points(dataset, i_algorithm, i) & get_points(dataset, j_algorithm, j)) > 0
            adjacency_matrix[i][j] *= i_silhouette + j_silhouette
    
    return adjacency_matrix

def clusters_from_bitstring(bitstring, dataset, clusters):
    mask = list(map(lambda x: x == '1', bitstring))
    selected_clusters = clusters[['label', 'algorithm']][mask].values.tolist()

    cumulative_df = []
    for label, name in selected_clusters:    
        selected_points = dataset.loc[dataset[name] == label, ['x', 'y']]
        selected_points['label'] = [label] * selected_points.shape[0]
        cumulative_df.append(selected_points)
    
    return (pd.concat(cumulative_df, ignore_index=True), selected_clusters)