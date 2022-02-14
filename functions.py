# Here, is the collection of functions used in the LDA project folder.
# TODO: mode this out, this is LDA functions

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import LatentDirichletAllocation
from tqdm import tqdm
from time import time
import zipfile
import os
from sklearn.metrics import accuracy_score

import plotFunctions as vis

def fitLDA(n_components, data):

    """
    Fit a signle LDA model
    :param: n_components: number of topics
    :param: data: samples_gene_readcounts input matrix for LDA modelling
    :return: samples_celltypes, geprofiles_cell, LDA
    """

    t0 = time()

    LDA = LatentDirichletAllocation(n_components=n_components)
    LDA.fit(data)

    # Doc_topic extraction
    samples_celltypes = LDA.transform(data)

    # Topic_word row and normalized (ge profile per cell) extraction
    celltypes_genes_readcounts = LDA.components_                                                         # raw
    geprofiles_celltypes = celltypes_genes_readcounts  / celltypes_genes_readcounts.sum(axis=1, keepdims=True)  # normalized ConcR

    # Perplexity
    perplexity = LDA.perplexity(data)
    print('Perplexity: ', perplexity)

    print('Finished using function fitLDA(), runtime (min): ', (time() - t0)/60)
    return samples_celltypes, geprofiles_celltypes, celltypes_genes_readcounts, LDA


def performTopicPerplexityAnalysis(data, max_topics=10):
    """
    Use one matrix in data (doc_words_counts)  to analyse Number of Topic vs. Perplexity (and Score)

    :param data:  doc_word matrix, in icgc it's samples_genes_readcounts matrix
    :param max_topics: maximum number of topics to investigate, default = 10
    :return: perplexity, score, samples_celltypes, ns
    """

    t0 = time()

    n_components = np.arange(1, max_topics, 1)    # Number of +- components (topics)

    perplexity = []
    score = []
    samples_celltypes = []
    for n in tqdm(n_components):
        LDA = LatentDirichletAllocation(n_components=n)
        LDA.fit(data)
        samples_celltypes0 = LDA.transform(data)
        samples_celltypes.append(samples_celltypes0)
        perplexity.append(LDA.perplexity(data))
        score.append(LDA.score(data))
        del LDA, samples_celltypes0
        pass

    print('Finished using function performTopicPerplexityAnalysis(), runtime (min): ', (time() - t0) / 60)
    return perplexity, score, samples_celltypes, n_components


def loadZipfileTovariable(path_to_zip):

    """
    Load zipped file to the variable in jupyter and ide, given the file is in nupmy format.
    :param: string, path to zip file e.g.:'samples_tissues_count.zip'
    :return: numpy array of data
    """

    basename = os.path.basename(path_to_zip) # TODO: this may fail in linux as per stackoverflow. Change this.
    filename = basename[:-4]
    #filename, file_extension = os.path.splitext(path1) # This doesn't work, filename outputs long path
    path_out_zip = os.getcwd()
    if os.path.isdir(path_out_zip):
        print('[INFO] Unziping...')
        with zipfile.ZipFile(path_to_zip, 'r') as zip_ref:
            zip_ref.extractall(path_out_zip)
    else:
        raise ValueError('Directory doesn\'t exist.')

    filename_full = filename + '.npy'
    assert os.path.exists(filename_full)
    data = np.load(filename_full)
    os.remove(filename_full)
    print('[INFO] Finished running loadZipfileTovariable(). Zip to variable tranferred.')

    return data

def loadZipfile(path_to_zip):

    """
    Take in a zip file path and output unzipped file, without loading.
    File must be in cwd (current working directory)
    Note: this function could be integrated with loadZipfileTovariable()
    :param path_to_zip:
    :return: unzipped file
    """

    path_out_zip = 'Unzipped_file'
    if os.path.isdir(path_out_zip):
        print('[INFO] Directory exists. Unziping...')
        with zipfile.ZipFile(path_to_zip, 'r') as zip_ref:
            zip_ref.extractall(path_out_zip)
    else:
        print('[INFO] Directory doesn\'t exist. Creating...')
        os.makedirs(path_out_zip)
        print('[INFO] Unziping...')
        with zipfile.ZipFile(path_to_zip, 'r') as zip_ref:
            zip_ref.extractall(path_out_zip)
    print('[INFO] Unzipping done')

    return


def findUniqueTopicMatching(topic_similarity):
    """
    Find unique minimum indices of each row for the topic matching, so as no dublicates in the indices, e.g. no [1,4], [2,4] etc
    TODO: I did a limited test, but a proper test should be done for this function
    :param: topic_similarity must be a matrix, where amounts of rows are ground truth n_components, amounts of columns inferred n_componenets.
            Rows n_components must be equal to number of topics (tissues or celltypes) selected. Number of columns may be more than that.
    :return: topic_matching_unique
    """

    n_components = np.shape(topic_similarity)[0]

    rows, cols, sorted_dic = [], [], []
    sorted_matrix = np.argsort(topic_similarity, axis=None)
    for i in range(len(sorted_matrix)):
        #ind = np.where(topic_similarity == topic_similarity.flatten()[sorted_matrix[i]])
        ind = np.unravel_index(sorted_matrix[i], topic_similarity.shape)
        rows.append(int(ind[0]))
        cols.append(int(ind[1]))
        sorted_dic.append([int(ind[0]), int(ind[1])])  # for debugging

    ordered, topic_matching_unique, all_items1, all_items2 = [], [], [], []
    for i in range(n_components):  # rows
        ind = rows.index(i)
        item1 = rows[ind]
        item2 = cols[ind]
        if item2 not in all_items2 and item1 not in all_items1:
            all_items1.append(item1)
            all_items2.append(item2)
            ordered.append([item1, item2])
            topic_matching_unique.append(item2)
        else:
            for j in range(ind + 1, len(cols)):
                item1_1 = rows[j]
                item2 = cols[j]
                if item2 not in all_items2 and item1_1 == item1:
                    all_items1.append(item1)
                    all_items2.append(item2)
                    ordered.append([item1, item2])
                    topic_matching_unique.append(item2)
                    break
                else:
                    continue

    topic_matching_unique = np.array(topic_matching_unique)
    print('Finished function findUniqueTopicMatching()')
    print(topic_matching_unique)

    return topic_matching_unique


def checkPresence(samples0, samples1):
    """
    Check if all samples0 are present in sample1
    :param:  2 lists of samples
    :returns: prints  'All samples are present' OR 'NOT all samples are present'
    """

    print('Checking if all samples0 are present in sample1, function checkPresence()')

    present = 0
    abscent = 0
    for i in samples0:
        if i in samples1:
            present = present + 1
        else:
            abscent = abscent + 1

    if abscent == 0:
        print('Results: all samples are present')
    else:
        print('Results: NOT all samples are present')
    del i, present, abscent

    return


def convertProbsToOnehot(doc_topic_dist):
    """
    Convert probabilities of each document from the continuous [0..1] values to the binary onehot [0 or 1] per document.
    Based on finding tha maximum probability in each row

    :param doc_topic_dist: matrix, documents on the rows, topics on the columns, probabilities [0..1]
    :return: onehot matrix, doc_topic_dist
    """

    doc_topic_dist_onehot= 0*(doc_topic_dist.copy())
    for i in range(np.shape(doc_topic_dist)[0]):
        items = doc_topic_dist[i, :]
        max_item_ind = np.argmax(items)
        doc_topic_dist_onehot[i, max_item_ind] = int(1)
    doc_topic_dist_onehot = doc_topic_dist_onehot.astype(int)

    return doc_topic_dist_onehot


def reverseOnehotEncoding(onehot_mat, tissues_selected):
    """
    Reverse one-hot encoded matrix to the 1D vector of labels

    :param onehot_mat: matrix with rows - samples, columns - class (tissue)
    :param tissues_selected: list of tissues as 1D vector
    :return: reversed vector of all classes
    """

    onehot_mat2 = np.argmax(onehot_mat, axis=1)[:, None]
    reversed_vec = []
    for i in onehot_mat2[:, 0]:
        tissue = tissues_selected[i]
        reversed_vec.append(tissue)
        del tissue
    del i

    reversed_vec = np.array(reversed_vec)
    reversed_vec = reversed_vec[:, None]

    return reversed_vec

