"""
Base IO code for all datasets
"""
import csv
from collections import namedtuple
from os import environ, makedirs
from os.path import expanduser, join
from pathlib import Path
from importlib import resources
import sys

import numpy as np

from ..utils import Jar  # Jar is imported inside __init__ file
from ..utils import check_pandas_support  # check_pandas_support is def inside __init__ file

DATA_MODULE = "core.datasets.data"

RemoteFileMetadata = namedtuple("RemoteFileMetadata", ["filename", "url"])


def get_data_home(data_home=None) -> str:
    if data_home is None:
        data_home = environ.get("HOTC_DATA", join(f"{Path.cwd()}", "/data"))
    data_home = expanduser(data_home)
    makedirs(data_home, exist_ok=True)
    return data_home


def get_glove_data_home(data_home=None) -> str:
    if data_home is None:
        data_home = environ.get("HOTC_DATA", join(f"{Path.cwd()}", "/glove_data"))
    data_home = expanduser(data_home)
    makedirs(data_home, exist_ok=True)
    return data_home


def _convert_data_dataframe(
        caller_name,
        data,
        target,
        feature_names,
        target_names,
        sparse_data=False
):
    pd = check_pandas_support("{} with as_frame=True".format(caller_name))
    if not sparse_data:
        data_df = pd.DataFrame(data, columns=feature_names, copy=False)
    else:
        data_df = pd.DataFrame.sparse.from_spmatrix(data, columns=feature_names)

    target_df = pd.DataFrame(target, columns=target_names)
    combined_df = pd.concat([data_df, target_df], axis=1)
    X = combined_df[feature_names]
    y = combined_df[target_names]
    if y.shape[1] == 1:
        y = y.iloc[:, 0]
    return combined_df, X, y


def load_csv_data(
        data_file_name,
        *,
        data_module=DATA_MODULE,
):
    """
    Loads `data_file_name` from `data_module with `importlib.resources`.
    Parameters
    ----------
    data_file_name : str
        Name of csv file to be loaded from `data_module/data_file_name`.
        For example `'bbc_data.csv'`.
    data_module : str or module, default='core.datasets.data'
        Module where data lives. The default is `'core.datasets.data'`.

    Returns
    -------
    data : ndarray of shape (n_samples, n_features)
        A 2D array with each row representing one sample and each column
        representing the features of a given sample.
    target : ndarray of shape (n_samples,)
        A 1D array holding target variables for all the samples in `data`.
        For example target[0] is the target variable for data[0].
    target_names : ndarray of shape (n_samples,)
        A 1D array containing the names of the classifications. For example
        target_names[0] is the name of the target[0] class.
    """

    csv_file = resources.files(data_module).joinpath(data_file_name).open("r") \
        if (sys.version_info >= (3, 9)) \
        else resources.path(data_module, data_file_name)

    data_file = csv.reader(csv_file)
    temp = next(data_file)
    n_samples = int(temp[0])
    n_features = int(temp[1])
    target_names = np.array(temp[2:])
    data = np.empty((n_samples, n_features))
    target = np.empty((n_samples,), dtype=int)

    for i, ir in enumerate(data_file):
        data[i] = np.asarray(ir[:-1], dtype=np.float64)
        target[i] = np.asarray(ir[-1], dtype=int)

    return data, target, target_names


def load_iris(*, return_X_y=False, as_frame=False):
    data_file_name = "iris.csv"
    data, target, target_names = load_csv_data(
        data_file_name=data_file_name
    )

    feature_names = [
        "sepal length (cm)",
        "sepal width (cm)",
        "petal length (cm)",
        "petal width (cm)",
    ]

    frame = None
    target_columns = [
        "target",
    ]
    if as_frame:
        frame, data, target = _convert_data_dataframe(
            "load_iris", data, target, feature_names, target_columns
        )

    if return_X_y:
        return data, target

    return Jar(
        data=data,
        target=target,
        frame=frame,
        target_names=target_names,
        feature_names=feature_names,
        filename=data_file_name,
        data_module=DATA_MODULE,
    )