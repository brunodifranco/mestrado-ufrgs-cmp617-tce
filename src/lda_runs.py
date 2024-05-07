import nltk
import os
import json
import pickle
from pathlib import Path
from logging import Logger
from typing import Tuple, List
import numpy as np
from gensim.corpora import Dictionary
from lda_fit import LDASingleModel
from utils.utils import logger, get_json
from evaluate_lda import coherence_score

nltk.download("punkt")


class LDARuns:
    """
    Runs the LDA model 20 times, to get the mean and std of the metrics c_uci, c_npmi, c_v, u_mass.
    It's important because of the LDA model variability.

    Attributes
    ----------
    params_json : Path
        JSON file path with model parameters.
    logger : Logger, defaults to logger
        logger.
    """

    def __init__(
        self,
        params_json: Path,
        logger: Logger = logger,
    ):

        self.params_json = params_json
        self.logger = logger

    def load_model_inputs(self) -> Tuple[List[str], List, Dictionary, LDASingleModel]:
        """
        Loads the model inputs: vec, corpus, id2word, lda

        Returns
        -------
        vec, corpus, id2word, lda: Tuple[List[str], List, Dictionary, LDASingleModel]
            Model and model inputs.
        """

        lda = LDASingleModel(
            params_json=self.params_json,
            model_name="any",  # set to any here, as the LDASingleModel won't be used to save the model
        )

        vec = lda.nlp_preprocessing()
        corpus, id2word = lda.create_corpus(vec)

        return vec, corpus, id2word, lda

    def run_models(
        self, vec: List[str], corpus: List, id2word: Dictionary, lda: LDASingleModel
    ):
        """
        Loads the model inputs: vec, corpus, id2word, lda

        Parameters
        ----------
        vec, corpus, id2word, lda: Tuple[List[str], List, Dictionary, LDASingleModel]
            Model and model inputs.
        """
        params_json = get_json(self.params_json)

        num_runs = (
            20  # Fixed at 20 runs, to keep the same amount of iterations for all models
        )
        topn = params_json["topn"]
        nlp_normalization_method = params_json["nlp_normalization_method"]
        metrics = []

        best_cv_score = float("-inf")  # Initially set to -inf
        best_model = None

        for i in range(num_runs):

            print(f"Now running run number {i}")
            lda_model = lda.fit(corpus, id2word)

            c_uci = coherence_score(lda_model, vec, id2word, "c_uci", topn)
            c_npmi = coherence_score(lda_model, vec, id2word, "c_npmi", topn)
            c_v = coherence_score(lda_model, vec, id2word, "c_v", topn)
            u_mass = coherence_score(lda_model, vec, id2word, "u_mass", topn)

            # Gets the best model based on "c_v"
            if c_v > best_cv_score:
                best_cv_score = c_v
                best_model = lda_model

            # Saving the metrics
            metrics.append(
                {
                    "run": i + 1,
                    "c_uci": c_uci,
                    "c_npmi": c_npmi,
                    "c_v": c_v,
                    "u_mass": u_mass,
                }
            )

        # Gets the mean and std for each metric
        mean_metrics = {
            "c_uci_mean": np.mean([m["c_uci"] for m in metrics]),
            "c_npmi_mean": np.mean([m["c_npmi"] for m in metrics]),
            "c_v_mean": np.mean([m["c_v"] for m in metrics]),
            "u_mass_mean": np.mean([m["u_mass"] for m in metrics]),
            "c_uci_std": np.std([m["c_uci"] for m in metrics]),
            "c_npmi_std": np.std([m["c_npmi"] for m in metrics]),
            "c_v_std": np.std([m["c_v"] for m in metrics]),
            "u_mass_std": np.std([m["u_mass"] for m in metrics]),
        }

        results = {"metrics": metrics, "mean_metrics": mean_metrics}

        # Saves the best model
        path = f"models/lda/{nlp_normalization_method}/best_model_topn_{topn}/"
        if not os.path.exists(path):
            os.makedirs(path + "pkl")
            os.makedirs(path + "metrics")

        pickle.dump(best_model, open(path + "pkl/model.pkl", "wb"))
        pickle.dump(vec, open(path + "pkl/vec.pkl", "wb"))
        pickle.dump(corpus, open(path + "pkl/corpus.pkl", "wb"))
        pickle.dump(id2word, open(path + "pkl/id2word.pkl", "wb"))

        self.logger.info(f"Best model and its files saved to {path}")

        # Saves the metrics mean and std
        with open(path + "metrics/lda_metrics.json", "w") as json_file:
            json.dump(results, json_file, indent=4)

        self.logger.info("Metrics saved!")

    def run(self):
        """Runs the pipeline"""

        self.logger.setLevel("INFO")
        vec, corpus, id2word, lda = self.load_model_inputs()

        self.logger.setLevel("WARNING")
        self.run_models(vec, corpus, id2word, lda)


if __name__ == "__main__":
    lda = LDARuns(
        params_json="src/lda_opt_outputs/results_lemmatization_topn_10.json",
    )
    lda.run()
