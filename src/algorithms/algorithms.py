from abc import ABC, abstractmethod
from typing import Optional, Union, Any
from numpy.typing import ArrayLike, NDArray #use this to match predict sig from the lib.

from src.dataset import print_dataset_details, preprocess_datasets, load_dataset

class BaseAlgorithm(ABC):
    """
    Abstract base class for all the algorithms in this project.
    Used to make the tool calls much easier.
    """

    def __init__(self) -> None:
        """Initialise the model with no data or internal state."""
        self.data = None  # placeholder for loaded dataset
        self.model = None # placeholder for the actual model

    def load_dataset(self, name: str) -> None:
        """
        Load a dataset from a file path or an identifier string (eg "caffe", "pytorch", etc)

        Parameters
        ----------
        name : str
            The name or path of the dataset to load.
        """
        self.data = load_dataset(name)
        self.project = name  # Store the project name for later use in model saving/loading

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
    def train(self, repetitions: int = 10, seed: Optional[int] = 51003) -> dict[str, Any]:
        """
        Train the model using the loaded data.

        Parameters
        ----------
        repetitions : int, default=10
            The number of times to repeat the training process.
        seed : int or None, default=51003
            Optional random seed for reproducibility.

        Returns
        -------
        metrics : dict
            A dictionary containing evaluation metrics (e.g., accuracy, precision) for the trained model on the test set over all repetitions.
        """
        raise NotImplementedError("Method 'train' not implemented.")

    @abstractmethod
    def predict(self, X: str) -> int:
        """
        Make predictions using the trained model.

        Parameters
        ----------
        X : str
            The input text to predict.

        Returns
        -------
        y_pred : int
            Predicted value. 0 for negative sentiment, 1 for positive sentiment (not a bug, a bug)
        """
        raise NotImplementedError("Method 'predict' not implemented.")