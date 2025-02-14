import os
import torch
import numpy as np
import pickle as pkl

MODULE_PATH = os.path.dirname(__file__)
DICTIONARY_PATH = os.path.join(MODULE_PATH, "data/dictionaries")


def load_dictionary_from_pickle(path):
    dictionary_file = open(path, "rb")
    numpy_dictionary = pkl.load(dictionary_file)
    dictionary_file.close()
    dictionary = torch.tensor(numpy_dictionary.astype(np.float32))
    return dictionary


def load_bars_dictionary():
    path = os.path.join(DICTIONARY_PATH, "bars", "bars-16_by_16.p")
    return load_dictionary_from_pickle(path)


def load_olshausen_dictionary():
    path = os.path.join(DICTIONARY_PATH, "olshausen", "olshausen-1.5x_overcomplete.p")
    return load_dictionary_from_pickle(path)
