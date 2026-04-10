from .algorithms import BaseAlgorithm
from .baseline.nb_classification import NBAlgorithm as BaselineAlgorithm
from .improved_1.svm_classification import SVMAlgorithm as Improved1Algorithm
from .improved_2.svm_classification import SVMAlgorithm as Improved2Algorithm
from .improved_3.rf_classification import RFAlgorithm as Improved3Algorithm
from .improved_4.svm_classification import SVMAlgorithm as Improved4Algorithm
from .improved_5.svm_classification import SVMAlgorithm as Improved5Algorithm
from .improved_6.bert_classification import BERTAlgorithm as Improved6Algorithm
from .final.svm_classification import SVMAlgorithm as FinalAlgorithm

# Gets the algorithm class based on the name, useful for CLI and replication code.
def get_algorithm(name: str) -> BaseAlgorithm:
    algorithms = {
        'baseline': BaselineAlgorithm,
        'base': BaselineAlgorithm,

        'improved_1': Improved1Algorithm,
        'improved-1': Improved1Algorithm,
        'improved1': Improved1Algorithm,

        'improved_2': Improved2Algorithm,
        'improved-2': Improved2Algorithm,
        'improved2': Improved2Algorithm,

        'improved_3': Improved3Algorithm,
        'improved-3': Improved3Algorithm,
        'improved3': Improved3Algorithm,

        'improved_4': Improved4Algorithm,
        'improved-4': Improved4Algorithm,
        'improved4': Improved4Algorithm,

        'improved_5': Improved5Algorithm,
        'improved-5': Improved5Algorithm,
        'improved5': Improved5Algorithm,

        'improved_6': Improved6Algorithm,
        'improved-6': Improved6Algorithm,
        'improved6': Improved6Algorithm,

        'final': FinalAlgorithm,
    }

    name = name.lower()
    
    if name not in algorithms:
        raise ValueError(f"Unknown algorithm name: {name}. Valid options are: {list(algorithms.keys())}")
    
    return algorithms[name]()