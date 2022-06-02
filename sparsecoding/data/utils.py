import os
import torch
import numpy as np
import pickle as pkl

MODULE_PATH = os.path.dirname(__file__)
DATASET_PATH = os.path.join(MODULE_PATH, "datasets")
DICTIONARY_PATH = os.path.join(MODULE_PATH, "dictionaries")

BARS_DICT_PATH = os.path.join(DICTIONARY_PATH, "bars", "bars-16_by_16.p")
OLSHAUSEN_DICT_PATH = os.path.join(DICTIONARY_PATH, "olshausen", "olshausen-1.5x_overcomplete.p")


def load_dictionary_from_pickle(path):
    dictionary_file = open(path, 'rb')
    numpy_dictionary = pkl.load(dictionary_file)
    dictionary_file.close()
    dictionary = torch.tensor(numpy_dictionary.astype(np.float32))
    return dictionary


def load_bars_dictionary():
    path = BARS_DICT_PATH
    dictionary_file = open(path, 'rb')
    numpy_dictionary = pkl.load(dictionary_file)
    dictionary_file.close()
    dictionary = torch.tensor(numpy_dictionary.astype(np.float32))
    return dictionary


def load_olshausen_dictionary():
    path = OLSHAUSEN_DICT_PATH
    dictionary_file = open(path, 'rb')
    numpy_dictionary = pkl.load(dictionary_file)
    dictionary_file.close()
    dictionary = torch.tensor(numpy_dictionary.astype(np.float32))
    return dictionary
