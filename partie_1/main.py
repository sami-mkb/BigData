# -*- coding: utf-8 -*-
"""
Created on Mon Dec 27 11:27:09 2021

@author: Hugo - Yacine - Sami
"""

import numpy
from scipy.io import arff
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.metrics import davies_bouldin_score
import time

from sklearn.preprocessing import StandardScaler


# files from the dataset artificial

path_smile1 = "./artificial/smile1.arff"
path_2d_3c_no123 = "./artificial/2d-3c-no123.arff"
path_2d_4c_no4 = "./artificial/2d-4c-no4.arff"
path_donut1 = "./artificial/donut1.arff"
path_long1 = "./artificial/long1.arff"
path_blobs = "./artificial/blobs.arff"
path_smile2 = "./artificial/smile2.arff"
path_2sp2glob = "./artificial/2sp2glob.arff"
path_diamond9= "./artificial/diamond9.arff"



##################################################################
# READ a data set (arff format)

# Parser un fichier de données au format arff
# datanp est un tableau (numpy) d'exemples avec pour chacun la liste 
# des valeurs des features




def erase_file(file_path):  # This function erase a figure in a repositery 
    file = open(file_path, "w")
    file.close()


def add_execution_time(file_path, section_name):  # Open a txt file and add execution time
    f = open(file_path, "a")
    f.write(section_name + "\n")
    f.close()

def plot_fig(x, y, labels, name): # This function plot and save the figures in a repositery
    plt.figure()
    plt.scatter(x, y, c=labels, marker='x')
    plt.savefig(name)
    


def execute_agglo(nbCluster, data_train, linkage, label): # execute agglomeratif algorithm
    tps1 = time.time()
    agglo = AgglomerativeClustering(nbCluster, linkage=linkage)
    agglo.fit(data_train)
    tps2 = time.time() - tps1
    
    ########################################################################
    # TESTER PARAMETRES METHODE ET RECUPERER autres métriques
    ########################################################################
    
    silhouette_coefficient = metrics.silhouette_score(data_train, agglo.labels_, metric='euclidean')  
    davies_coefficient = davies_bouldin_score(data_train, agglo.labels_)  
    
        
    f = open("./timing_analysis/agglo_clustering/agglo_clustering.txt", "a")
    execution_time = "Temps d'execution " + label + " = %f\n" % tps2
    silhouette_time = "Coefficient de Silhouette = %f\n" % silhouette_coefficient
    time_davies = "Coefficient de Davies = %f\n" % davies_coefficient
    
    f.write(execution_time + silhouette_time + time_davies)
    f.close()
    plt.scatter(nbCluster, silhouette_coefficient, c='red', marker='.')
    plt.scatter(nbCluster, davies_coefficient, c='blue', marker='x')
    
    
    return agglo


def iter_Agglo(data, linkage, name):  # iterative function from 2 to 10
    add_execution_time("./timing_analysis/agglo_clustering/agglo_clustering.txt", "Agglomeratif Clustering [" + name +
                   "] [" + linkage + "]")
    plt.figure()
    for iter_cluster in range(2, 10):
        execute_agglo(iter_cluster, data, linkage,
                            "Agglomeratif " + linkage + "- " + name + " [" + str(iter_cluster) + "] [" + linkage + "]")
    plt.savefig("./metrics/agglomeratif/" + name)


def Save_plot_Agglo(nbCluster, data, linkage, name, x, y, name_fig): # save figures
    agglo = execute_agglo(nbCluster, data, linkage,
                                "Agglomeratif " + linkage + "- " + name + " [" + str(
                                    nbCluster) + "] [" + linkage + "]")
    plot_fig(x, y, agglo.labels_, "./figures_agglo/" + name_fig)


# Run  agglomerative clustering methods for a given number of clusters

def Agglo_Clustering(): # execute Agglomeratif algorithm for the datasets
    
    erase_file("./timing_analysis/agglo_clustering/agglo_clustering.txt")
    add_execution_time("./timing_analysis/agglo_clustering/agglo_clustering.txt", "Agglomeratif Clustering 2d-4c-no4"
                   + " nombre clusters fixé")

    data_2d4c = arff.loadarff(open(path_2d_4c_no4, 'r'))
    _2d4cno4 = np.array(data_2d4c, dtype=object)[0]
    _2d4cno4_train = list(zip(_2d4cno4['a0'], _2d4cno4['a1']))
    nbCluster = 4

    Save_plot_Agglo(nbCluster, _2d4cno4_train, "single", "2dc4", _2d4cno4['a0'], _2d4cno4['a1'], "2dc4_single")
    Save_plot_Agglo(nbCluster, _2d4cno4_train, "average", "2dc4", _2d4cno4['a0'], _2d4cno4['a1'], "2dc4_average")
    Save_plot_Agglo(nbCluster, _2d4cno4_train, "complete", "2dc4", _2d4cno4['a0'], _2d4cno4['a1'], "2dc4_complete")
    Save_plot_Agglo(nbCluster, _2d4cno4_train, "ward", "2dc4", _2d4cno4['a0'], _2d4cno4['a1'], "2dc4_ward")

    iter_Agglo(_2d4cno4_train, "single", "2d4c")
    iter_Agglo(_2d4cno4_train, "average", "2d4c")
    iter_Agglo(_2d4cno4_train, "complete", "2d4c")
    iter_Agglo(_2d4cno4_train, "ward", "2d4c")

    add_execution_time("./timing_analysis/agglo_clustering/agglo_clustering.txt", "Agglomeratif Clustering blobs"
                    + " nombre clusters fixé")
# Another example

    data_blobs = arff.loadarff(open(path_blobs, 'r'))
    _blobs = np.array(data_blobs, dtype=object)[0]
    _blobs_train = list(zip(_blobs['x'], _blobs['y']))
    nbCluster = 3

    Save_plot_Agglo(nbCluster, _blobs_train, "single", "blobs", _blobs['x'], _blobs['y'], "blobs_single")
    Save_plot_Agglo(nbCluster, _blobs_train, "average", "blobs", _blobs['x'], _blobs['y'], "blobs_average")
    Save_plot_Agglo(nbCluster, _blobs_train, "complete", "blobs", _blobs['x'], _blobs['y'], "blobs_complete")
    Save_plot_Agglo(nbCluster, _blobs_train, "ward", "blobs", _blobs['x'], _blobs['y'], "blobs_ward")

    iter_Agglo(_blobs_train, "single", "blobs")
    iter_Agglo(_blobs_train, "average", "blobs")
    iter_Agglo(_blobs_train, "complete", "blobs")
    iter_Agglo(_blobs_train, "ward", "blobs")
    

def execute_kmeans(nbCluster, data, label): # run kMeans algorithm


    tps1 = time.time()
    kmeans = KMeans(n_clusters=nbCluster, init='k-means++')
    kmeans.fit(data)
    
    tps2 = time.time() - tps1
    silhouette_coefficient = metrics.silhouette_score(data, kmeans.labels_, metric='euclidean')  
    davies_coefficient = davies_bouldin_score(data, kmeans.labels_)  
    
    
    f = open("./timing_analysis/kmeans_clustering/kmeans_clustering.txt", "a")
    execution_time = "Execution time " + label + " = %f\n" % tps2
    silhouette_time = "Coefficient de Silhouette = %f\n" % silhouette_coefficient
    time_davies = "Coefficient de Davies Bouldin = %f\n" % davies_coefficient
    
    
    f.write(execution_time + silhouette_time + time_davies)
    f.close()
    plt.scatter(nbCluster, silhouette_coefficient, c='red', marker='.')
    plt.scatter(nbCluster, davies_coefficient, c='blue', marker='x')
    return kmeans


def Save_plot_Kmeans(nbCluster, data, x, y, name, name_fig): # run algorithm and save the figures
    kmeans = execute_kmeans(nbCluster, data, "Kmeans- " + name + " [" + str(nbCluster) + "]")
    plot_fig(x, y, kmeans.labels_, "./figures_kmeans/" + name_fig)


def iter_KMeans(data, name): #iterative functions for kMeans
    add_execution_time("./timing_analysis/kmeans_clustering/kmeans_clustering.txt", "KMeans- " + name)
    plt.figure()
    for iter_cluster in range(2, 12):
        
        
        execute_kmeans(iter_cluster, data, "Kmeans- " + name + " [" + str(iter_cluster) + "]")
    plt.savefig("./metrics/kmeans/" + name)


##################################################################
# Run clustering K-Means method for a given number of clusters

def KMeans_Clustering(): # execute K-Means algorithm for datasets

    erase_file("./timing_analysis/kmeans_clustering/kmeans_clustering.txt")
    add_execution_time("./timing_analysis/kmeans_clustering/kmeans_clustering.txt", "KMeans 2d-4c-no4"
                    + " nombre clusters fixé")


    data_2d4c = arff.loadarff(open(path_2d_4c_no4, 'r'))
    _2d4cno4 = np.array(data_2d4c, dtype=object)[0]
    _2d4cno4_train = list(zip(_2d4cno4['a0'], _2d4cno4['a1']))
    
    
    nbCluster = 4

    plot_fig(_2d4cno4['a0'], _2d4cno4['a1'], _2d4cno4['class'], "./figures_kmeans/2d-4c-no4")
    # Résultat du clustering
    Save_plot_Kmeans(nbCluster, _2d4cno4_train, _2d4cno4['a0'], _2d4cno4['a1'], "2d-4c-no4", "2d-4c-no4_kmeans")
    iter_KMeans(_2d4cno4_train, "2d-4c-no4")

    add_execution_time("./timing_analysis/kmeans_clustering/kmeans_clustering.txt", "KMeans long1"
                    + " nombre clusters fixé")
# Another example

    data_long1 = arff.loadarff(open(path_long1, 'r'))
    _long1 = np.array(data_long1, dtype=object)[0]
    _long1_train = list(zip(_long1['a0'], _long1['a1']))
    nbCluster = 2

    plot_fig(_long1['a0'], _long1['a1'], _long1['class'], "./figures_kmeans/long1")
    Save_plot_Kmeans(nbCluster, _long1_train, _long1['a0'], _long1['a1'], "long1", "long1_kmeans")
    iter_KMeans(_long1_train, "long1")

    add_execution_time("./timing_analysis/kmeans_clustering/kmeans_clustering.txt", "KMeans blobs"
                    + " nombre clusters fixé")
# Another example
    data_blobs = arff.loadarff(open(path_blobs, 'r'))
    _blobs = np.array(data_blobs, dtype=object)[0]
    _blobs_train = list(zip(_blobs['x'], _blobs['y']))
    nbCluster = 3

    plot_fig(_blobs['x'], _blobs['y'], _blobs['class'], "./figures_kmeans/blobs")
    Save_plot_Kmeans(nbCluster, _blobs_train, _blobs['x'], _blobs['y'], "blobs", "blobs_kmeans")
    iter_KMeans(_blobs_train, "blobs")

    add_execution_time("./timing_analysis/kmeans_clustering/kmeans_clustering.txt", "KMeans smile2"
                    + " nombre clusters fixé")
# Another example

    smile2_data = arff.loadarff(open(path_smile2, 'r'))
    _smile2 = np.array(smile2_data, dtype=object)[0]
    smile2_train = list(zip(_smile2['a0'], _smile2['a1']))
    nbCluster = 4

    plot_fig(_smile2['a0'], _smile2['a1'], _smile2['class'], "./figures_kmeans/smile2")
    Save_plot_Kmeans(nbCluster, smile2_train, _smile2['a0'], _smile2['a1'], "smile2", "smile2_kmeans")
    iter_KMeans(smile2_train, "smile2")
    
    add_execution_time("./timing_analysis/kmeans_clustering/kmeans_clustering.txt", "KMeans diamond9"
                    + " nombre clusters fixé")
# Another example
    
    diamond9_data = arff.loadarff(open(path_diamond9, 'r'))
    _diamond9 = np.array(diamond9_data, dtype=object)[0]
    diamond9_train = list(zip(_diamond9['x'],_diamond9['y']))
    nbCluster = 9

    plot_fig(_diamond9['x'], _diamond9['y'], _diamond9['class'], "./figures_kmeans/_diamond9")
    Save_plot_Kmeans(nbCluster, diamond9_train, _diamond9['x'], _diamond9['y'], "diamond9", "_diamond9_kmeans")
    iter_KMeans(diamond9_train, "diamond9")



def execute_DBSCAN(distance, min_pts, data_train, label): # run DBSCAN algorithm 
    tps1 = time.time()
    data_train = StandardScaler().fit_transform(data_train)
    dbscan = DBSCAN(eps=distance, min_samples=min_pts).fit(data_train)
    tps2 = time.time() - tps1
    labels = dbscan.labels_
    
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)
    nbr_clusters = 'Estimated number of clusters: %d' % n_clusters_
    msg_noise = 'Estimated number of noise points: %d' % n_noise_
    f = open("./timing_analysis/dbscan_clustering/dbscan_clustering.txt", "a")
    execution_time = "Execution time " + label + " = %f\n" % tps2
    
    f.write(execution_time + nbr_clusters + "\n" + msg_noise + "\n")
    f.close()
    return dbscan


def Save_plot_DBSCAN(distance, min_pts, data, name, x, y, name_fig): # run and save DBSCAN figures 
    dbscan = execute_DBSCAN(distance, min_pts, data, "DBSCAN - " + name + " - dt[" + str(distance)
                                  + "] pts[" + str(min_pts) + "]")
    plot_fig(x, y, dbscan.labels_, "./figures_dbscan/" + name_fig)
    print("./figures_dbscan/" + name_fig)


def iter_DBSCANClustering(data, name, x, y): #iterative function for dbscan
    add_execution_time("./timing_analysis/dbscan_clustering/dbscan_clustering.txt", "DBSCAN Clustering [" + name + "]")
    for distance in numpy.linspace(0.1, 1, 10):
        print(distance)
        for samples in range(2, 10):
            distance = round(distance, 1)
            Save_plot_DBSCAN(distance, samples, data, name, x, y, name + "_" + "dt[" + str(distance).replace('.', ',')
                                + "]_pts[" + str(samples) + "]")

########################################################################
# Run DBSCAN clustering method 
# for a given number of parameters eps and min_samples
#  

def DBSCAN_Clustering(): # execute DBSCAN algorithm for datasets

    erase_file("./timing_analysis/dbscan_clustering/dbscan_clustering.txt")
    add_execution_time("./timing_analysis/dbscan_clustering/dbscan_clustering.txt", "DBSCAN 2d-4c-no4"
                   + " distance et nombre de points fixés")

    data_2d4c = arff.loadarff(open(path_2d_4c_no4, 'r'))
    _2d4cno4 = np.array(data_2d4c, dtype=object)[0]
    _2d4cno4_train = list(zip(_2d4cno4['a0'], _2d4cno4['a1']))
    distance = 5
    min_pts = 0.5

# Plot results
    Save_plot_DBSCAN(distance, min_pts, _2d4cno4_train, "2dc4", _2d4cno4['a0'], _2d4cno4['a1'], "2dc4_dbscan")


    add_execution_time("./timing_analysis/dbscan_clustering/dbscan_clustering.txt", "DBSCAN blobs"
                   + " distance et nombre de points fixés")
# Another example

    data_blobs = arff.loadarff(open(path_blobs, 'r'))
    _blobs = np.array(data_blobs, dtype=object)[0]
    _blobs_train = list(zip(_blobs['x'], _blobs['y']))
    distance = 0.35
    min_pts = 14

    Save_plot_DBSCAN(distance, min_pts, _blobs_train, "blobs", _blobs['x'], _blobs['y'], "blobs_dbscan")

    #iter_DBSCANClustering(_blobs_train, "blobs", _blobs['x'], _blobs['y'])

    add_execution_time("./timing_analysis/dbscan_clustering/dbscan_clustering.txt", "DBSCAN smile2"
                   + " distance et nombre de points fixés")
# Another example
    smile2_data = arff.loadarff(open(path_smile2, 'r'))
    _smile2 = np.array(smile2_data, dtype=object)[0]
    smile2_train = list(zip(_smile2['a0'], _smile2['a1']))
    distance = 5
    min_pts = 0.5

    Save_plot_DBSCAN(distance, min_pts, smile2_train, "smile2", _smile2['a0'], _smile2['a1'], "smile2_dbscan")

    #iter_DBSCANClustering(smile2_train, "smile2", _smile2['a0'], _smile2['a1'])


def main():
    
    KMeans_Clustering() 
    #Agglo_Clustering()
    #DBSCAN_Clustering()


if __name__ == "__main__":
    main()
