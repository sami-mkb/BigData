import numpy
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.metrics import davies_bouldin_score
import time
from sklearn.preprocessing import StandardScaler
import re

d32 =  "./new-data/d32.txt"
d64 = "./new-data/d64.txt"
n1 = "./new-data/n1.txt"
n2 = "./new-data/n2.txt"
w2 = "./new-data/w2.txt"


def erase_file(file_path): # This function erase a figure in a repositery 
    file = open(file_path, "w")
    file.close()

def plot_fig(x, y, labels, name):   # This function plot and save the figures in a repositery
    plt.figure()
    plt.scatter(x, y, c=labels, marker='.', s=0.5)
    plt.savefig(name)


def add_execution_time(file_path, section_name): # Open a txt file and add execution time
    f = open(file_path, "a")
    f.write(section_name + "\n")
    f.close()


def plot_fig_with_no_labels(x, y, name):
    plt.figure()
    plt.scatter(x, y, marker='.', s=0.5)
    plt.savefig(name)


def extract_data(filename): #extract data from the txt file
    data = []
    with open(filename, 'r') as f:
        for line in f.readlines():
            data.append(re.sub(' +', ' ', line.strip()).split(' '))
    return np.array(data, float)


def show_fig(): #visualize and sava the data representation
    
    
    d32_train = extract_data(d32)
    d64_train = extract_data(d64)
    n1_train = extract_data(n1)
    n2_train = extract_data(n2)
    w2_train = extract_data(w2)
    
    
    plot_fig_with_no_labels(d32_train[:, 0], d32_train[:, 1], './figures/d32')
    plot_fig_with_no_labels(d64_train[:, 0], d64_train[:, 1], './figures/d64')
    plot_fig_with_no_labels(n1_train[:, 0], n1_train[:, 1], './figures/n1')
    plot_fig_with_no_labels(n2_train[:, 0], n2_train[:, 1], './figures/n2')
    plot_fig_with_no_labels(w2_train[:, 0], w2_train[:, 1], './figures/w2')
    
    


def execute_kmeans(NbClusters, data, label):
    tps1 = time.time()
    kmeans = KMeans(n_clusters=NbClusters, init='k-means++')
    kmeans.fit(data)
    
    tps2 = time.time() - tps1
    
    
    ########################################################################
    # TESTER PARAMETRES METHODE ET RECUPERER autres métriques
    ########################################################################
    
    silhouette_coefficient = metrics.silhouette_score(data, kmeans.labels_, metric='euclidean') 
    indice_davies = davies_bouldin_score(data, kmeans.labels_)  
    f = open("./timing_analysis/kmeans_clustering/kmeans_clustering.txt", "a")
    
    ########################################################################
    # Execution time
    ########################################################################
    
    
    execution_time = "Temps d'execution " + label + " = %f\n" % tps2
    silhouette_time = "Coefficient de Silhouette = %f\n" % silhouette_coefficient
    davies_time = "Coefficient de Davies = %f\n" % indice_davies
    f.write(execution_time + silhouette_time + davies_time)
    f.close()
    plt.scatter(NbClusters, silhouette_coefficient, c='red', marker='.')
    plt.scatter(NbClusters, indice_davies, c='blue', marker='x')
    return kmeans


def Save_plot_Kmeans(NbClusters, data, x, y, name, name_fig):
    kmeans =  execute_kmeans(NbClusters, data, "Kmeans- " + name + " [" + str(NbClusters) + "]")
    plot_fig(x, y, kmeans.labels_, "./figures_kmeans/" + name_fig)



def execute_agglo(NbClusters, data_train, linkage, label):
    tps1 = time.time()
    agglo = AgglomerativeClustering(NbClusters, linkage=linkage)
    agglo.fit(data_train)
    tps2 = time.time() - tps1
    
    silhouette_coefficient = metrics.silhouette_score(data_train, agglo.labels_, metric='euclidean')  
    
    
    indice_davies = davies_bouldin_score(data_train, agglo.labels_) 
    f = open("./timing_analysis/agglo_clustering/agglo_clustering.txt", "a")
    execution_time = "Temps d'execution " + label + " = %f\n" % tps2
    silhouette_time = "Coefficient de Silhouette = %f\n" % silhouette_coefficient
    
    davies_time = "Coefficient de Davies = %f\n" % indice_davies
    f.write(execution_time + silhouette_time + davies_time)
    
    f.close()
    plt.scatter(NbClusters, silhouette_coefficient, c='red', marker='.')
    plt.scatter(NbClusters, indice_davies, c='blue', marker='x')
    return agglo


def iter_AggloClustering(data, linkage, name):
    
    add_execution_time("./timing_analysis/agglo_clustering/agglo_clustering.txt", "Agglomeratif Clustering [" + name +
                   "] [" + linkage + "]")
    plt.figure()
    for iter_cluster in range(2, 10):
        execute_agglo(iter_cluster, data, linkage,
                            "Agglomeratif " + linkage + "- " + name + " [" + str(iter_cluster) + "] [" + linkage + "]")
    
    plt.savefig("./metrics/agglomeratif/" + name)


def Save_plot_Agglo(NbClusters, data, linkage, name, x, y, name_fig):
    agglo = execute_agglo(NbClusters, data, linkage,
                                "Agglomeratif " + linkage + "- " + name + " [" + str(
                                    NbClusters) + "] [" + linkage + "]")
    plot_fig(x, y, agglo.labels_, "./figures_agglo/" + name_fig)


def runClustering_Agglomeratif(NbClusters, file, name):
    data = extract_data(file)
    Save_plot_Agglo(NbClusters, data, "single", name, data[:, 0], data[:, 1], name + "_single")
    Save_plot_Agglo(NbClusters, data, "average", name, data[:, 0], data[:, 1], name + "_average")
    
    
    Save_plot_Agglo(NbClusters, data, "complete", name, data[:, 0], data[:, 1], name + "_complete")
    Save_plot_Agglo(NbClusters, data, "ward", name, data[:, 0], data[:, 1], name + "_ward")
    
    
    iter_AggloClustering(data, "single", name)
   
    iter_AggloClustering(data, "average", name)
    iter_AggloClustering(data, "complete", name)
    iter_AggloClustering(data, "ward", name)


def Agglo_Clusterin():
    
    # erase_file("./timing_analysis/agglo_clustering/agglo_clustering.txt")
    # add_execution_time("./timing_analysis/agglo_clustering/agglo_clustering.txt", "Agglomeratif Clustering d32"
    #                + " nombre clusters fixé")

    NbClusters = 5
    # runClustering_Agglomeratif(NbClusters, d32, "d32")

    # add_execution_time("./timing_analysis/agglo_clustering/agglo_clustering.txt", "Agglomeratif Clustering d64"
    #                + " nombre clusters fixé")

    # runClustering_Agglomeratif(NbClusters, d64, "d64")

    # add_execution_time("./timing_analysis/agglo_clustering/agglo_clustering.txt", "Agglomeratif Clustering n1"
    #                + " nombre clusters fixé")

    # runClustering_Agglomeratif(NbClusters, n1, "n1")

    add_execution_time("./timing_analysis/agglo_clustering/agglo_clustering.txt", "Agglomeratif Clustering n2"
                    + " nombre clusters fixé")

    runClustering_Agglomeratif(NbClusters, w2, "w2")
    

def iter_KMeansClustering(data, name):
    add_execution_time("./timing_analysis/kmeans_clustering/kmeans_clustering.txt", "KMeans- " + name)
    plt.figure()
    
    for iter_cluster in range(5, 20):
         execute_kmeans(iter_cluster, data, "Kmeans- " + name + " [" + str(iter_cluster) + "]")
    
    plt.savefig("./metrics/kmeans/" + name)


def runClustering_KMeans(data, name):
    data_train = extract_data(data)
    
    NbClusters = 16
    plot_fig_with_no_labels(data_train[:, 0], data_train[:, 1], "./figures_kmeans/" + name)
    Save_plot_Kmeans(NbClusters, data_train, data_train[:, 0], data_train[:, 1], name, name + "_kmeans")
    iter_KMeansClustering(data_train, name)


def KMeans_Clustering():
    erase_file("./timing_analysis/kmeans_clustering/kmeans_clustering.txt")
    
    # add_execution_time("./timing_analysis/kmeans_clustering/kmeans_clustering.txt", "KMeans d32"
    #                + " nombre clusters fixé")

    # print("d32")
    # runKMeans_Clustering(d32, "d32")
   

    add_execution_time("./timing_analysis/kmeans_clustering/kmeans_clustering.txt", "KMeans d64"
                    + " nombre clusters fixé")
    print("d64")
    runClustering_KMeans(d64, "d64")

    # add_execution_time("./timing_analysis/kmeans_clustering/kmeans_clustering.txt", "KMeans n1"
    #                + " nombre clusters fixé")
    # print("n1")
    # runClustering_KMeans(n1, "n1")

    # add_execution_time("./timing_analysis/kmeans_clustering/kmeans_clustering.txt", "KMeans n2"
    #                + " nombre clusters fixé")
    # print("n2")
    # runClustering_KMeans(n2, "n2")
    
    # add_execution_time("./timing_analysis/kmeans_clustering/kmeans_clustering.txt", "KMeans y1"
    #                + " nombre clusters fixé")

    # print("y1")
    # runClustering_KMeans(y1, "y1")





def execute_DBSCAN(distance, min_pts, data_train, label):
    tps1 = time.time()
    data_train = StandardScaler().fit_transform(data_train)
    dbscan = DBSCAN(eps=distance, min_samples=min_pts).fit(data_train)
    
    tps2 = time.time() - tps1
    labels = dbscan.labels_
    # Number of clusters in labels, ignoring noise if present.
    
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)
    msg_clusters = 'Estimated number of clusters: %d' % n_clusters_
    msg_noise = 'Estimated number of noise points: %d' % n_noise_
    f = open("./timing_analysis/dbscan_clustering/dbscan_clustering.txt", "a")
    execution_time = "Temps d'execution " + label + " = %f\n" % tps2
   
    f.write(execution_time + msg_clusters + "\n" + msg_noise + "\n")
    f.close()
    return dbscan


def Save_plot_DBSCAN(distance, min_pts, data, name, x, y, name_fig):
    dbscan = execute_DBSCAN(distance, min_pts, data, "DBSCAN - " + name + " - dt[" + str(distance)
                                  + "] pts[" + str(min_pts) + "]")
    plot_fig(x, y, dbscan.labels_, "./figures_dbscan/" + name_fig)
    print("./figures_dbscan/" + name_fig)


def iter_DBSCANClustering(data, name, x, y):
    add_execution_time("./execution_time/dbscan_clustering/dbscan_clustering.txt", "DBSCAN Clustering [" + name + "]")
    
    for distance in numpy.linspace(0.1, 0.2, 20):
        print(distance)
        for samples in range(2, 10):
            distance = round(distance, 1)
            Save_plot_DBSCAN(distance, samples, data, name, x, y, name + "_" + "dt[" + str(distance).replace('.', ',')
                              + "]_pts[" + str(samples) + "]")


def runClustering_DBSCAN(filename, distance, min_pts, name):
    data_train = extract_data(filename)
    Save_plot_DBSCAN(distance, min_pts, data_train, name, data_train[:, 0], data_train[:, 1], name)


def DBSCAN_Clustering():
    erase_file("./execution_time/dbscan_clustering/dbscan_clustering.txt")
    
    add_execution_time("./execution_time/dbscan_clustering/dbscan_clustering.txt", "DBSCAN d32"
                   + " distance et nombre de points fixés")
    distance = 0.5
    min_pts = 14
    runClustering_DBSCAN(d32, distance, min_pts, "d32")
    #data = extract_data(d32)
    #iter_DBSCANClustering(data, "d32", data[:, 0], data[:, 1])

    add_execution_time("./execution_time/dbscan_clustering/dbscan_clustering.txt", "DBSCAN d64"
                   + " distance et nombre de points fixés")

    distance = 0.35
    min_pts = 14
    runClustering_DBSCAN(d64, distance, min_pts, "d64")
    #data = extract_data(d64)
    #iter_DBSCANClustering(data, "d32", data[:, 0], data[:, 1])

    add_execution_time("./execution_time/dbscan_clustering/dbscan_clustering.txt", "DBSCAN n1"
                   + " distance et nombre de points fixés")

    distance = 5
    min_pts = 0.5
    runClustering_DBSCAN(n1, distance, min_pts, "n1")
    #data = extract_data(n1)
    #iter_DBSCANClustering(data, "n1", data[:, 0], data[:, 1])
    
    add_execution_time("./execution_time/dbscan_clustering/dbscan_clustering.txt", "DBSCAN n2"
                   + " distance et nombre de points fixés")
    
    distance = 0.0891
    min_pts = 20
    runClustering_DBSCAN(n2, distance, min_pts, "n2")
    #data = extract_data(n2)
    #iter_DBSCANClustering(data, "n2", data[:, 0], data[:, 1])
    


def main():

    #show_fig()

    KMeans_Clustering()

    #Agglo_Clusterin()

    #DBSCAN_Clustering()



if __name__ == "__main__":
    main()
