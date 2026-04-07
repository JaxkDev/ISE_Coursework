from abc import ABC, abstractmethod
from typing import Optional, Union, Any
from numpy.typing import ArrayLike, NDArray #use this to match predict sig from the lib.

import numpy as np
import pandas as pd

# Other Algorithms
# from src.algorithms.baseline.nb_classification import NBAlgorithm as BaselineAlgorithm
# from src.algorithms.improved_1.svm_classification import SVMAlgorithm as Improved1Algorithm
# from src.algorithms.improved_2.svm_classification import SVMAlgorithm as Improved2Algorithm
# from src.algorithms.improved_3.rf_classification import RFAlgorithm as Improved3Algorithm
# from src.algorithms.improved_4.svm_classification import SVMAlgorithm as Improved4Algorithm
# from src.algorithms.improved_5.svm_classification import SVMAlgorithm as Improved5Algorithm
# from src.algorithms.improved_6.bert_classification import BERTAlgorithm as Improved6Algorithm
#...

from src.dataset import print_dataset_details, preprocess_datasets

class BaseAlgorithm(ABC):
    """
    Abstract base class for all the algorithms in this project.
    Used to make the tool calls much easier.
    """

    def __init__(self) -> None:
        """Initialise the model with no data or internal state."""
        self.data = pd.DataFrame()  # placeholder for loaded dataset
        self.model = None           # placeholder for the actual model

    @abstractmethod
    def load_dataset(self, name: str) -> None:
        """
        Load a dataset from a file path or an identifier string (eg "caffe", "pytorch", etc)

        Parameters
        ----------
        name : str
            The name or path of the dataset to load.
        """
        raise NotImplementedError("Method 'load_dataset' not implemented.")

    @abstractmethod
    def load_data(self, raw_csv: Any) -> None:
        """
        Load data from a raw CSV object (e.g., file-like object or string).

        Parameters
        ----------
        raw_csv : file-like object or str
            The raw CSV content to parse and load.
        """
        raise NotImplementedError("Method 'load_data' not implemented.")
    
    @abstractmethod
    def preprocess_data(self) -> None:
        """
        Perform any necessary preprocessing on the loaded data.
        """
        raise NotImplementedError("Method 'preprocess_data' not implemented.")

    @abstractmethod
    def load_model(self, fresh: bool = True) -> None:
        """
        Load or initialise the internal model.

        Parameters
        ----------
        fresh : bool, default=True
            If True, create a new empty model.
            If False, load a previously saved model from disk.
        """
        raise NotImplementedError("Method 'load_model' not implemented.")

    @abstractmethod
    def pre_train(self, seed: Optional[int] = None) -> None:
        """
        Perform any necessary preprocessing or unsupervised pre-training.

        Parameters
        ----------
        seed : int or None, default=None
            Optional random seed for reproducibility.
        """
        raise NotImplementedError("Method 'pre_train' not implemented.")

    @abstractmethod
    def train(self, seed: Optional[int] = None) -> None:
        """
        Train the model using the loaded data.

        Parameters
        ----------
        seed : int or None, default=None
            Optional random seed for reproducibility.
        """
        raise NotImplementedError("Method 'train' not implemented.")

    @abstractmethod
    def predict(self, X: ArrayLike) -> NDArray[Any]:
        """
        Make predictions using the trained model.

        This signature matches the common scikit‑learn / numpy predict
        interface, accepting any array-like input and returning a numpy ndarray.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,) or (n_samples, n_outputs)
            Predicted values.
        """
        raise NotImplementedError("Method 'predict' not implemented.")