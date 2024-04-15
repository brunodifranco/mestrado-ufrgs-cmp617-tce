import os
import pickle
import json
from lda_fit import LDASingleModel

class LDARuns:
    def __init__(
        self,
        params_json: Path,
        model_name: str,
        logger: Logger = logger,
    ):

        self.params_json = get_json(params_json)
        self.model_name = model_name
        self.logger = logger

    def load_model_inputs(self):



        lda = LDASingleModel(
        params_json=self.model_params_json,
        model_name="any", # set to any here, as the LDASingleModel won't be used to save the model
        )

        vec = lda.nlp_preprocessing()
        corpus, id2word = lda.create_corpus(vec)


        return vec, corpus, id2word, lda


# Defina seu número de runs
num_runs = 20

# Inicialize uma lista para armazenar todas as métricas
metrics = []

# Inicialize a melhor métrica de c_v como negativa infinita para começar
best_cv_score = float('-inf')
best_model = None

for i in range(num_runs):

    print(f"Now running run number {i}")
    lda_model = lda.fit(corpus, id2word)
    
    c_uci = coherence_score(lda_model, vec, id2word, "c_uci", topn)
    c_npmi = coherence_score(lda_model, vec, id2word, "c_npmi", topn)
    c_v = coherence_score(lda_model, vec, id2word, "c_v", topn)
    u_mass = coherence_score(lda_model, vec, id2word, "u_mass", topn)
    
    # Avalie se c_v é melhor que a melhor até agora
    if c_v > best_cv_score:
        best_cv_score = c_v
        best_model = lda_model
    
    # Armazene as métricas desta iteração
    metrics.append({
        "run": i+1,
        "c_uci": c_uci,
        "c_npmi": c_npmi,
        "c_v": c_v,
        "u_mass": u_mass
    })
    
# Calcule a média e o desvio padrão das métricas
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

# Salve as métricas e o melhor modelo em JSON
results = {
    "metrics": metrics,
    "mean_metrics": mean_metrics
}

# Salve o melhor modelo
path = f"models/lda/best_model_topn_{topn}/"
if not os.path.exists(path):
    os.makedirs(path)

pickle.dump(best_model, open(path + "model.pkl", "wb"))
pickle.dump(vec, open(path + "vec.pkl", "wb"))
pickle.dump(corpus, open(path + "corpus.pkl", "wb"))
pickle.dump(id2word, open(path + "id2word.pkl", "wb"))

print(f"Best model and its files saved to {path}")

# Salve as métricas em um arquivo JSON
with open("lda_metrics.json", "w") as json_file:
    json.dump(results, json_file, indent=4)

print("Metrics saved to lda_metrics.json")
