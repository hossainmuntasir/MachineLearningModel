from pathlib import Path
from abc import ABC, abstractmethod
from typing import Callable, Any, Self
import pandas as pd
import numpy as np
from sklearn.model_selection import ParameterGrid
from sklearn.base import BaseEstimator
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, matthews_corrcoef, average_precision_score
from joblib import Parallel, delayed
from tqdm import tqdm

class GridSearchModel(ABC):
    """
    Abstract base class that uses grid search to find the best model.
    """
    OUTPUT_FOLDER = Path("output")
    SCORING = {
        "accuracy": lambda y_val, predictions: accuracy_score(y_val, predictions["predict"]),
        "f1": lambda y_val, predictions: f1_score(y_val, predictions["predict"], pos_label="On"),
        "roc_auc": lambda y_val, predictions: roc_auc_score(y_val, predictions["proba"]),
        "average_precision": lambda y_val, predictions: average_precision_score(y_val, predictions["proba"], pos_label="On"),
        "matthews": lambda y_val, predictions: matthews_corrcoef(y_val, predictions["predict"]),
    }

    def __init__(self, save_path: str | Path = None, random_state=42, n_jobs = 1):
        self._results = None
        self._random_state = random_state
        self._n_jobs = n_jobs
        self._model = self._base(random_state=42)

        self._output_folder = self.OUTPUT_FOLDER
        if save_path is not None:
            self._output_folder = self._output_folder / save_path
        self._index_file = self._output_folder / "index.pkl"
        # create output folder if it does not exist
        self._output_folder.mkdir(parents=True, exist_ok=True)

    @property
    @abstractmethod
    def _base(self) -> type[BaseEstimator]:
        pass

    @property
    @abstractmethod
    def _extra_columns(self) -> dict[str, Callable[[BaseEstimator], Any]]:
        pass

    @property
    def model(self):
        return self._model

    @property
    def results(self):
        if self._results is None:
            return None
        return pd.DataFrame(self._results)

    def fit(self, x_train, y_train, x_val, y_val, show_progress=True, reset=False) -> Self:
        # load previous results
        param_grid = ParameterGrid(self._search_params(x_train, y_train))
        self._results = self._read_index_file(param_grid)
        if reset:
            for v in self._results.values():
                v.clear()

        params_to_check = []
        for params in param_grid:
            if params not in self._results["params"]:
                params_to_check.append(params)

        models = self._models_from_param_grid(params_to_check, x_train, y_train, x_val, y_val, 
                                              n_jobs=self._n_jobs, show_progress=show_progress)
        for params, clf, predictions, scores in models:
            self._update_index_file(clf, params, predictions, scores)

        # remove empty columns
        rows = len(self._results["params"])
        keys = list(self._results.keys())
        for k in keys:
            if len(self._results[k]) != rows:
                del self._results[k]

        # NOTE: check which score to use?
        best = np.argmax(self._results["f1"])
        params = self._results["params"][best]
        self._model.set_params(**params)
        self._model.fit(x_train, y_train)

        return self

    def save_results(self):
        self.results.to_csv(self._output_folder / "results.csv")

    def save_model(self):
        # TODO: save trained model to file
        pass

    @abstractmethod
    def _search_params(self, x_train, y_train) -> dict[str, Any]:
        """Returns the search parameters to be used for creating the models to be compared."""
        pass

    def _models_from_param_grid(self, param_grid: dict[str, Any], x_train, y_train, x_val, y_val, n_jobs=-1, show_progress=True):
        if len(param_grid) == 0:
            return
        
        base = self._base

        def clf(params):
            # train model
            model = base(random_state=self._random_state, **params)
            model.fit(x_train, y_train)

            predictions = {
                "predict": model.predict(x_val),
                "proba": model.predict_proba(x_val)[:,1],
            }

            scores = {k: func(y_val, predictions) for k, func in self.SCORING.items()}
            return params, model, predictions, scores

        #parallel = Parallel(n_jobs=-1, verbose=show_progress, return_as="generator_unordered")
        parallel = Parallel(n_jobs=n_jobs, return_as="generator_unordered")
        if show_progress:
            #print(f"Need to complete {len(param_grid)} tasks.")
            param_grid = tqdm(param_grid)
        yield from parallel(delayed(clf)(params) for params in param_grid)

        #for params in param_grid:
        #    clf = self._base(random_state=self._random_state, **params)
        #    clf.fit(x_train, y_train)
        #    yield params, clf

    def _read_index_file(self, param_grid):
        """Try to load previously computed data from file."""
        default = {
            "params": [], 
            #"filename": [],
        }
        for column in param_grid[0].keys():
            default[column] = []
        for column in self._extra_columns:
            default[column] = []
        for column in self.SCORING.keys():
            default[column] = []

        if not self._index_file.is_file():
            # no file found
            return default

        result = pd.read_pickle(self._index_file)
        if not all(p in param_grid for p in result["params"]):
            # search parameters are different
            return default
        elif not all(c in result for c in default.keys()):
            # column names are different
            return default
        return result

    def _update_index_file(self, clf, params, predictions, scores):
        #n = str(len(self._results["params"]))
        #filename = f"val{n}.pkl"
        #pd.to_pickle(predictions, self._output_folder / filename)

        self._results["params"].append(params)
        #self._results["filename"].append(filename)
        for k, v in params.items():
            self._results[k].append(v)
        for k, func in self._extra_columns.items():
            self._results[k].append(func(clf))
        for k, v in scores.items():
            self._results[k].append(v)
        pd.to_pickle(self._results, self._index_file)

class GridSearchDecisionTree(GridSearchModel):
    OUTPUT_FOLDER = Path("output/decision_tree")

    def __init__(self, save_path: str | Path = None, random_state=42, n_jobs=1):
        super().__init__(save_path, random_state, n_jobs)

    @property
    def _base(self):
        return DecisionTreeClassifier

    @property
    def _extra_columns(self) -> dict[str, Callable[[BaseEstimator], Any]]:
        return {
            "depth": lambda clf: clf.tree_.max_depth,
            "nodes": lambda clf: clf.tree_.node_count,
        }

    def _search_params(self, x_train, y_train):
        """Returns the search parameters to be used for creating the models to be compared."""
        self._model.set_params(ccp_alpha = 0.0)
        self._model.fit(x_train, y_train)
        ccp_alphas = self._model.cost_complexity_pruning_path(x_train, y_train).ccp_alphas
        return {"ccp_alpha": np.unique(ccp_alphas)}
