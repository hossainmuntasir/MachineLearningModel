from pathlib import Path
from abc import ABC, abstractmethod
from typing import Callable, Any, Self, Literal
from itertools import chain
import time
import pandas as pd
import numpy as np
from sklearn.model_selection import ParameterGrid
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, matthews_corrcoef, average_precision_score
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder, TargetEncoder
from joblib import Parallel, delayed
from tqdm import tqdm

class GridSearchModel(ABC):
    """Abstract base class that uses grid search to find the best model.

    Parameters
    ----------
    save_path : str, Path
                Folder in which to save results
    n_fits : int, default=1
                number of models to train in parallel
    **base_params : dict, optinal
                    extra keyword parameters will be passed through to the trained models

    If `random_state` is not specified, it will be set to a default integer value.
    """
    OUTPUT_FOLDER = Path("output")
    SCORING = {
        "accuracy": lambda y_val, predictions: accuracy_score(y_val, predictions["predict"]),
        "f1": lambda y_val, predictions: f1_score(y_val, predictions["predict"], pos_label="On"),
        "roc_auc": lambda y_val, predictions: roc_auc_score(y_val, predictions["proba"]),
        "pr_auc": lambda y_val, predictions: average_precision_score(y_val, predictions["proba"], pos_label="On"),
        "matthews": lambda y_val, predictions: matthews_corrcoef(y_val, predictions["predict"]),
    }

    def __init__(self, save_path: str | Path = None, n_fits = 1, **base_params):
        self._results = None
        self._n_fits = n_fits

        self._base_params = base_params
        if "random_state" not in base_params:
            self._base_params["random_state"] = 42
        self._random_state = base_params["random_state"]
        self._model = self._base(**self._base_params)
        self._model = make_pipeline(*self._transform_layers, self._model)

        self._output_folder = self.OUTPUT_FOLDER
        self._index_file = self._output_folder / "index.pkl"
        if save_path is not None:
            self._output_folder = self._output_folder / save_path
            self._index_file = self.OUTPUT_FOLDER / f"{save_path}.pkl"
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
    @abstractmethod
    def _best_params_filter(self) -> dict[str, Literal["min", "max"]]:
        """If multiple models end up with the same score, these filters are used to select the best one.
        """
        pass

    @property
    def model(self):
        return self._model

    @property
    def results(self):
        if self._results is None:
            return None
        return pd.DataFrame(self._results)

    @property
    def _transform_layers(self) -> list[TransformerMixin]:
        # by default, returns a list with only one layer that encodes `Zone_name`
        return [
            make_column_transformer(
                (OneHotEncoder(), ["Zone_name"]), 
                remainder="passthrough"
            )
        ]

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

        if len(params_to_check) == 0:
            if show_progress:
                print("All parameters have been previously checked.")
        else:
            # fit and apply transformation layers
            x_train_transformed, x_val_transformed = x_train, x_val
            for layer in self._transform_layers:
                layer.fit(x_train_transformed, y_val)
                x_train_transformed = layer.transform(x_train_transformed)
                x_val_transformed = layer.transform(x_val_transformed)

            models = self._models_from_param_grid(params_to_check, x_train_transformed, y_train, x_val_transformed, y_val, 
                                                  n_jobs=self._n_fits, show_progress=show_progress)
            for params, clf, predictions, scores in models:
                self._update_index_file(clf, params, predictions, scores)

            # remove empty columns
            rows = len(self._results["params"])
            keys = list(self._results.keys())
            for k in keys:
                if len(self._results[k]) != rows:
                    del self._results[k]
            
        # NOTE: check which score to use?
        params = self.best_params("f1")
        self._model[-1].set_params(**params)
        if self._n_fits != 1 and "n_jobs" in self._model[-1].get_params() and "n_jobs" not in self._base_params:
            n_jobs = self._model[-1].get_params()["n_jobs"]
            self._model[-1].set_params(n_jobs=self._n_fits)
            self._model.fit(x_train, y_train)
            self._model[-1].set_params(n_jobs=n_jobs)
        else:
            self._model.fit(x_train, y_train)

        return self
    
    def best_params(self, scoring = "f1"):
        if self.results is None:
            return

        best: float = self.results[scoring].max()
        results = self.results[self.results[scoring] == best]

        for column, min_max in self._best_params_filter.items():
            if len(results) == 1:
                break
            if min_max == "min":
                best = results[column].min()
            elif min_max == "max":
                best = results[column].max()
            else:
                continue
            results = results[results[column] == best]

        return results["params"].iloc[0]
    
    def save_results(self, filepath = None):
        if len(self._results["params"]) == 0:
            return

        results = self.results
        columns = results.columns.to_list()
        columns.remove("params")

        if filepath is None:
            filepath = self._output_folder
        else:
            filepath = Path(filepath)
        filepath.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filepath = filepath / f"{timestamp}.csv"

        results.to_csv(filepath, columns=columns, index=False)
        return filepath

    def save_model(self):
        # TODO: save trained model to file
        pass

    @abstractmethod
    def _search_params(self, x_train, y_train) -> dict[str, Any]:
        """Returns the search parameters to be used for creating the models to be compared.
        """
        pass

    def _models_from_param_grid(self, param_grid: list[dict[str, Any]], x_train, y_train, x_val, y_val, n_jobs=-1, show_progress=True):
        if len(param_grid) == 0:
            return

        if show_progress:
            #print(f"Need to complete {len(param_grid)} tasks.")
            param_grid = tqdm(param_grid)

        base = self._base
        def clf(params):
            # train model
            model = base(**self._base_params, **params)
            model.fit(x_train, y_train)

            predictions = {
                "predict": model.predict(x_val),
                "proba": model.predict_proba(x_val)[:,1],
            }

            scores = {k: func(y_val, predictions) for k, func in self.SCORING.items()}
            return params, model, predictions, scores

        if n_jobs == 1:
            for params in param_grid:
                yield clf(params)
        else:
            #parallel = Parallel(n_jobs=-1, verbose=show_progress, return_as="generator_unordered")
            parallel = Parallel(n_jobs=n_jobs, return_as="generator_unordered")
            yield from parallel(delayed(clf)(params) for params in param_grid)

    def _read_index_file(self, param_grid):
        """Try to load previously computed data from file.
        """
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

    def __init__(self, save_path: str | Path = None, n_fits=1, **base_params):
        super().__init__(save_path, n_fits, **base_params)

    @property
    def _base(self):
        return DecisionTreeClassifier

    @property
    def _extra_columns(self) -> dict[str, Callable[[BaseEstimator], Any]]:
        return {
            "depth": lambda clf: clf.tree_.max_depth,
            "nodes": lambda clf: clf.tree_.node_count,
        }

    @property
    def _best_params_filter(self) -> dict[str, Literal["min", "max"]]:
        return {
            "ccp_alpha": "max"
        }

    def _search_params(self, x_train, y_train):
        self._model[-1].set_params(ccp_alpha = 0.0)
        self._model.fit(x_train, y_train)
        x_train = self._model[0].transform(x_train)
        ccp_alphas = self._model[-1].cost_complexity_pruning_path(x_train, y_train).ccp_alphas
        return {"ccp_alpha": np.unique(ccp_alphas)}

class GridSearchRandomForest(GridSearchModel):
    OUTPUT_FOLDER = Path("output/random_forest")

    def __init__(self, save_path: str | Path = None, n_fits=1, **base_params):
        super().__init__(save_path, n_fits, **base_params)

    @property
    def _base(self):
        return RandomForestClassifier

    @property
    def _extra_columns(self) -> dict[str, Callable[[BaseEstimator], Any]]:
        return {}
    
    @property
    def _best_params_filter(self) -> dict[str, Literal["min", "max"]]:
        return {
            "max_features": "min",
            "min_samples_split": "max",
            "n_estimators": "min",
        }

    @property
    def _transform_layers(self) -> list[TransformerMixin]:
        # We are using TargetEncoder for random forest to reduce the search time
        return [
            make_column_transformer(
                (TargetEncoder(random_state=self._random_state), ["Zone_name"]), 
                remainder="passthrough"
            )
        ]

    def fit(self, x_train, y_train, x_val, y_val, show_progress=True, reset=False) -> Self:
        # for random forest, we can use the `warm_start` parameter to reuse previous `n_estimators` values
        n_estimators = [50, 100, 150, 200, 300]

        # load previous results
        search_params = self._search_params(x_train, y_train)
        param_grid = ParameterGrid(search_params | {"n_estimators": n_estimators})
        self._results = self._read_index_file(param_grid)
        if reset:
            for v in self._results.values():
                v.clear()

        params_to_check = []
        param_grid = ParameterGrid(search_params)
        for params in param_grid:
            # check that all values of n_estimators have results for a given params configuration
            if not all(params | {"n_estimators": n} in self._results["params"] for n in n_estimators):
                params_to_check.append(params)

        if len(params_to_check) == 0:
            if show_progress:
                print("All parameters have been previously checked.")
        else:
            # fit and apply transformation layers
            x_train_transformed, x_val_transformed = x_train, x_val
            for layer in self._transform_layers:
                layer.fit(x_train_transformed, y_train)
                x_train_transformed = layer.transform(x_train_transformed)
                x_val_transformed = layer.transform(x_val_transformed)

            models = self._models_from_param_grid(params_to_check, x_train_transformed, y_train, x_val_transformed, y_val, 
                                                  n_estimators, n_jobs=self._n_fits, show_progress=show_progress)
            for params, clf, predictions, scores in models:
                self._update_index_file(clf, params, predictions, scores)

            # remove empty columns
            rows = len(self._results["params"])
            keys = list(self._results.keys())
            for k in keys:
                if len(self._results[k]) != rows:
                    del self._results[k]

        # NOTE: check which score to use?
        params = self.best_params("f1")
        self._model[-1].set_params(**params)
        if self._n_fits != 1 and "n_jobs" not in self._base_params:
            n_jobs = self._model[-1].get_params()["n_jobs"]
            self._model[-1].set_params(n_jobs=self._n_fits)
            self._model.fit(x_train, y_train)
            self._model[-1].set_params(n_jobs=n_jobs)
        else:
            self._model.fit(x_train, y_train)

        return self

    def _search_params(self, x_train: pd.DataFrame, y_train: pd.Series):
        n_features = len(x_train.columns)

        return {
            #"n_estimators": [100, 200, 300], # see self.fit()
            "max_features": range(1, n_features),
            "min_samples_split": range(2, 11),
            "criterion": ["gini", "entropy", "log_loss"]
        }

    def _models_from_param_grid(self, param_grid: list[dict[str, Any]], x_train, y_train, x_val, y_val, 
                                n_estimators = [100, 200, 300], n_jobs=-1, show_progress=True):
        if len(param_grid) == 0:
            return

        if show_progress:
            #print(f"Need to complete {len(param_grid)} tasks.")
            param_grid = tqdm(param_grid)

        base = self._base
        def clf(params):
            results = []

            # train model
            model = base(**self._base_params, **params, warm_start=True)
            for n in n_estimators:
                model.set_params(n_estimators=n)
                model.fit(x_train, y_train)

                predictions = {
                    "predict": model.predict(x_val),
                    "proba": model.predict_proba(x_val)[:,1],
                }

                scores = {k: func(y_val, predictions) for k, func in self.SCORING.items()}
                results.append((params | {"n_estimators": n}, model, predictions, scores))
            return results

        if n_jobs == 1:
            for params in param_grid:
                yield from clf(params)
        else:
            #parallel = Parallel(n_jobs=-1, verbose=show_progress, return_as="generator_unordered")
            parallel = Parallel(n_jobs=n_jobs, return_as="generator_unordered")
            yield from chain.from_iterable(parallel(delayed(clf)(params) for params in param_grid))

SCORES = tuple(GridSearchModel.SCORING.keys())
