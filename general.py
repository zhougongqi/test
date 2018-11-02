import numpy as np
import math
import sys

import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


def data_shuffle(data: np.ndarray) -> np.ndarray:
    """
    shuffle 2-D data along rows
    """
    idx = np.arange(len(data))
    np.random.shuffle(idx)
    data = data[idx, :]
    return data


def split_data(data: np.ndarray, proportion: float):
    """
    split colomn direction into two parts: train and test. 
    """
    assert 0 < proportion < 1, "proportion must be in (0,1)"
    row, col = data.shape
    pp_idx = int(row * proportion)
    train = data[0:pp_idx, :]
    test = data[pp_idx:row, :]
    return train, test


def strip_labels(data: np.ndarray):
    row, col = data.shape
    feat = data[:, 0 : col - 1]
    label = data[:, col - 1]
    return feat, label.astype(np.int8)


def stat_labels(label: np.ndarray):
    ul = np.unique(label)
    for i in range(len(ul)):
        ni = len(label[label == ul[i]])
        print("class {} : {}".format(i, ni))


def reduce_feature(data, bands):
    row, col = data.shape
    icol = col / bands
    d = data[0, :]
    d = d.reshape(bands, -1)
    ds = d[:, 0::15]
    ds = ds.flatten()
    dout = ds
    for r in range(1, row):
        d = data[r, :]
        d = d.reshape(bands, -1)
        ds = d[:, 0::15]
        ds = ds.flatten()
        dout = np.vstack((dout, ds.flatten()))
        print_progress_bar(r, row)
    print(" ")
    print("reduced shape:", dout.shape)
    return dout


def reduce_feature_ndvi(data, bands):
    row, col = data.shape
    icol = col / bands
    d = data[0, :]
    d = d.reshape(bands, -1)
    ds = d[:, 0::7]
    ds = (ds[3, :] - ds[2, :]) / (ds[3, :] + ds[2, :])
    ds = ds.flatten()
    dout = ds
    for r in range(1, row):
        d = data[r, :]
        d = d.reshape(bands, -1)
        ds = d[:, 0::7]
        ds = (ds[3, :] - ds[2, :]) / (ds[3, :] + ds[2, :])
        ds = ds.flatten()
        dout = np.vstack((dout, ds.flatten()))
        print_progress_bar(r, row)
    print(" ")
    print("reduced shape:", dout.shape)
    return dout


def delete_label(data, useless_labels: list):
    d = data
    row, col = data.shape
    for ul in useless_labels:
        idx_nan = np.where(d[:, -1] == ul)
        if len(idx_nan):
            d = np.delete(d, idx_nan, axis=0)
            # l = data[r, col - 1]
            # if l in useless_labels:
            #     d = np.delete(d, r, 0)
    return d


def print_progress_bar(now_pos: int, total_pos: int):
    n_sharp = math.floor(50 * now_pos / total_pos)
    n_space = 50 - n_sharp
    sys.stdout.write(
        "  ["
        + "#" * n_sharp
        + " " * n_space
        + "]"
        + "{:.2%}\r".format(now_pos / total_pos)
    )


def cm_plot(y, yp):
    cm = confusion_matrix(y, yp)
    plt.matshow(cm, cmap=plt.cm.Greens)
    plt.colorbar()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    for x in range(len(cm)):
        for y in range(len(cm)):
            plt.annotate(
                cm[x, y],
                xy=(x, y),
                horizontalalignment="center",
                verticalalignment="center",
            )
    return plt


def get_average(list0):
    lsum = 0
    for item in list0:
        lsum += item
    return lsum / len(list0)
