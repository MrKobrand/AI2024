import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as shc


def plot_dendrogram(linkage_matrix):
    plt.figure(figsize=(10, 7))
    plt.title("Dendrograms")
    dend = shc.dendrogram(linkage_matrix)
    plt.show()
