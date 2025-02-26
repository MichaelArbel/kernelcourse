from kernels.base_kernels import Kernel
import numpy as np

import networkx as nx


class WLKernel(Kernel):
    """
    The Weisfeler-Lehman subraph kernel.
    """
    def __init__(self, iterations=3, optimized=False):
        super().__init__()
        self.iterations = iterations
        self.optimized = optimized

    def weisfeiler_lehman_subgraph_hashes(self, G):
        """
        Computes the WL subgraph hashes.
        :param G: A networkx graph.
        :return: A list of list of hashes (one list for each node).
        """
        # Initialize labels
        labels = {node: str(G.nodes[node]['labels']) for node in G.nodes()}

        final_hashes = {node: list() for node in G.nodes()}

        for _ in range(self.iterations):
            new_labels = {}
            for node in G.nodes():
                # Compute new features
                label_list = [str(G.edges[node,nbr]['labels']) + labels[nbr] for nbr in G.neighbors(node)]
                label = labels[node] + "".join(sorted(label_list))

                # Hash features
                hashed_label = str(hash(label))

                new_labels[node] = hashed_label
                final_hashes[node].append(hashed_label)

            labels = new_labels

        return final_hashes

    def kernel(self, X, Y):
        # Input lists of graphs X and Y of len N and M

        X_hash = [sorted(
            [subgraph_hash for vertex_hash in self.weisfeiler_lehman_subgraph_hashes(G).values() for subgraph_hash in
             vertex_hash]) for G in X]
        Y_hash = [sorted(
            [subgraph_hash for vertex_hash in self.weisfeiler_lehman_subgraph_hashes(G).values() for subgraph_hash in
             vertex_hash]) for G in Y]
        if self.optimized:
            K = optimized_coocurrence_matrix(X_hash, Y_hash)
        else:
            K = cooccurrence_matrix(X_hash, Y_hash)

        return K








def count_coocurrences(list1, list2, normalized=True):
    """
    Count the number of matching elements in two sorted lists
    """
    count = 0
    i, j = 0, 0
    N, M = len(list1), len(list2)

    while i < N and j < M:
        if list1[i] == list2[j]:
            count += 1
            i += 1
            j += 1
        elif list1[i] < list2[j]:
            i += 1
        else:
            j += 1

    if normalized:
        return 2*count/(len(list1) + len(list2))
    else:
        return count


def cooccurrence_matrix(array1, array2):
    """
    Compute the co-occurence matrix between the two arrays.
    array1: list of lists
    array2: list of lists
    return: np.array of shape (len(array1), len(array2))
    """
    K, L = len(array1), len(array2)

    mesh1, mesh2 = np.meshgrid(np.array(array1), np.array(array2))

    M = [count_coocurrences(i, j) for (i, j) in zip(mesh1.flatten(), mesh2.flatten())]

    M = np.array(M).reshape(L, K).transpose()

    return np.array(M)


def create_hash_dict(array):
    """
    From an array of list of hashes, create a dictionary of lists that links each hash to the indices containing it in
    the array, as well as the nb of occurences in this array element.

    :param array: list of lists
    :return: dictionary of list of tuples. Each key is a hash sequence, the tuples (i, cur_count, array_length)
    correspond to the array index containing the hash, and cur_count is the nb of occurences of this hash in the array
    element i. array_length stores the length of the list contained in the ith element of the array, and used for
    normalization purpose.
    """
    hash_dict = dict()
    for i in range(len(array)):
        hashes, counts = np.unique(array[i], return_counts=True)
        for cur_hash, cur_count in zip(hashes, counts):
            if cur_hash not in hash_dict:
                hash_dict[cur_hash] = [(i, cur_count, len(array[i]))]
            else:
                hash_dict[cur_hash].append((i, cur_count, len(array[i])))

    return hash_dict


def get_hash_coocurrences(list, hash_dict, vector_size):
    """
    Computes the co-occurrence vector between the input and all the lists appearing in the global variable
    hash_dict.
    :param list: A list of hashes
    :return: The co-occurrence vector of the list and the lists of the (corresponding to one row of the co-occurrence
    matrix)0).
    """
    count_vector = np.zeros(vector_size)
    hashes, counts = np.unique(list, return_counts=True)
    for cur_hash, cur_count in zip(hashes, counts):
        if cur_hash in hash_dict:
            for i, hash_count, list_length in hash_dict[cur_hash]:
                count_vector[i] += 2 * min(cur_count, hash_count) / (len(list) + list_length)

    return count_vector


def optimized_coocurrence_matrix(array1, array2):
    """
    Computes the co-occurence matrix between array1 and array2, using a dict of list to avoid redundant computations.
    :param array1: list of lists
    :param array2: list of lists
    :return: np.array of shape (len(array1), len(array2))
    """

    # Stores the hash occurrences of array2 in a dictionary
    hash_dict = create_hash_dict(array2)

    vect_list = [get_hash_coocurrences(cur_list, hash_dict, len(array2)) for cur_list in array1]

    K = np.vstack(vect_list)

    return K











