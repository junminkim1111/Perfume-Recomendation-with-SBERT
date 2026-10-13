from sentence_transformers import SentenceTransformer
import time
import pandas as pd
import torch
from torch import nn
import random
from functions import get_IDF_weights


device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
print(f"USING DEVICE: {device}")

# Load dataset / model / hyperparameters
model = SentenceTransformer("all-MiniLM-L6-v2").to(device)

dataset = pd.read_csv("fra_cleaned.csv", sep=";", encoding="unicode_escape", on_bad_lines="skip")

IDF_weight = get_IDF_weights(dataset)

layer_weights = torch.tensor([[0.8, 0.25, 0.1],
                               [0.25, 1., 0.35],
                               [0.1, 0.35, 1.2]], device=device)
layer_weights = layer_weights / torch.sum(layer_weights)


class PreProcess:
    def __init__(self, model, device, dataset, IDF_weights=None):
        self.model = model
        self.dataset = dataset
        self.device = device
        self.note_embedding_cache = {}
        self.IDF_weight = IDF_weights if IDF_weights is not None else {note: 1. for note in IDF_weights.keys()}

    def get_note_embedding(self, note):
        if note not in self.note_embedding_cache:
            self.note_embedding_cache[note] = self.model.encode(
                note, convert_to_tensor=True, device=self.device
            )
        return self.note_embedding_cache[note]

    def process_notes(self, note_dict):
        processed_notes = {}
        for k, v in note_dict.items():
            temp = [n.strip() for n in v.split(",")]
            weight = torch.tensor([self.IDF_weight[w] for w in temp]).unsqueeze(1)
            weight = weight / torch.sum(weight)
            weight = weight.to(self.device)
            embeddings = torch.stack([self.get_note_embedding(n) for n in temp])
            processed_notes[k] = torch.sum(embeddings * weight, dim=0)

        weighted_embeddings = torch.vstack(list(processed_notes.values()))
        return weighted_embeddings  # shape = (3, 384)

    def get_emb_dict(self):
        emb_DB = {}
        for idx in range(len(self.dataset)):
            name = self.dataset.at[idx, "Perfume"]
            notes = {
                "Top": self.dataset.at[idx, "Top"],
                "Middle": self.dataset.at[idx, "Middle"],
                "Base": self.dataset.at[idx, "Base"],
            }
            note_embeddings = self.process_notes(notes)
            emb_DB[name] = note_embeddings
        return emb_DB

    def dict_to_tensor(self, emb_DB):
        names = list(emb_DB.keys())
        tensors = list(emb_DB.values())
        embeddings = torch.stack(tensors)
        embeddings = embeddings.to(self.device)
        return names, embeddings


class Perfume_Recommender(PreProcess):
    def __init__(self, dataset, model, device, layer_weights, DB_batch, names, IDF_weights=None):
        super().__init__(model, device, dataset, IDF_weights=IDF_weights)
        self.layer_weights = layer_weights
        self.DB_batch = DB_batch
        self.names = names

    def timer(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            print(f"Duration: {end_time - start_time:.4f} seconds")
            return result
        return wrapper

    @timer
    def recommend_idx(self, query_idx="random", n_recommendations=5):
        if query_idx == "random":
            query_idx = random.randint(0, len(self.dataset) - 1)

        target_name = self.dataset.at[query_idx, "Perfume"]
        target_notes = {
            "Top": self.dataset.at[query_idx, "Top"],
            "Middle": self.dataset.at[query_idx, "Middle"],
            "Base": self.dataset.at[query_idx, "Base"],
        }
        weighted_embeddings_target = self.process_notes(target_notes)

        q_norm = nn.functional.normalize(weighted_embeddings_target, p=2, dim=1)
        db_norm = nn.functional.normalize(self.DB_batch, p=2, dim=2)

        similarity_map = torch.einsum("id, bjd -> bij", q_norm, db_norm)
        scores = (similarity_map * self.layer_weights).sum(dim=(1, 2))
        top_scores, top_indices = torch.topk(scores, k=n_recommendations + 1)
        return target_notes, target_name, top_scores, top_indices

    def recommend(self, query_notes, n_recommendations=5):
        weighted_embeddings_target = self.process_notes(query_notes)

        q_norm = nn.functional.normalize(weighted_embeddings_target, p=2, dim=1)
        db_norm = nn.functional.normalize(self.DB_batch, p=2, dim=2)

        similarity_map = torch.einsum("id, bjd -> bij", q_norm, db_norm)
        scores = (similarity_map * self.layer_weights).sum(dim=(1, 2))
        top_scores, top_indices = torch.topk(scores, k=n_recommendations + 1)
        return query_notes, top_scores, top_indices

    def eval(self):
        cnt = 0
        wrong_list = []
        for query_idx in range(len(self.dataset)):
            target_name = self.dataset.at[query_idx, "Perfume"]
            target_notes = {
                "Top": self.dataset.at[query_idx, "Top"],
                "Middle": self.dataset.at[query_idx, "Middle"],
                "Base": self.dataset.at[query_idx, "Base"],
            }
            _, _, top_indices = self.recommend(target_notes, n_recommendations=1)
            top_idx = top_indices.tolist()
            top_similars = [self.names[i] for i in top_idx]
            if target_name == top_similars[0]:
                cnt += 1
            else:
                wrong_list.append(target_name)
        accuracy = cnt / len(self.dataset)
        return accuracy, wrong_list

    def display_recommendations(self, target_notes, target_name, top_scores, top_indices, note_comparison=True):
        top_idx = top_indices.tolist()
        top_similars = [self.names[i] for i in top_idx]

        if target_name in top_similars:
            top_similars.remove(target_name)
        else:
            top_similars = top_similars[:-1]

        print("***" * 10, "results", "***" * 10)
        brand_match = self.dataset.loc[self.dataset["Perfume"].str.strip() == target_name, "Brand"]
        brand = brand_match.values[0] if len(brand_match) > 0 else "Unknown Brand"
        print(f">>Original Perfume: {target_name} @ {brand}")
        similar_brand = self.dataset.loc[self.dataset["Perfume"].str.strip() == top_similars[0], "Brand"].values[0]
        print(f">>most similar perfume: {top_similars[0]} @ {similar_brand}")
        print("---" * 10)
        print(">>Top 5 similar perfumes:")
        for i, perfume in enumerate(top_similars):
            print(f"NO.{i+1} {perfume} / Similarity Score: {top_scores[i].item():.2f}")
        print()
        if note_comparison:
            print("---" * 10, "\n")
            print(f">>Most Similar Perfume except Original Perfume: {top_similars[0]}")
            print(f"Top Notes: {self.dataset.loc[self.dataset['Perfume'].str.strip() == top_similars[0], 'Top'].values[0]}")
            print(f"Middle Notes: {self.dataset.loc[self.dataset['Perfume'].str.strip() == top_similars[0], 'Middle'].values[0]}")
            print(f"Base Notes: {self.dataset.loc[self.dataset['Perfume'].str.strip() == top_similars[0], 'Base'].values[0]}")
            print("---" * 10)
            print(f">>Original Perfume: {target_name}")
            for k, v in target_notes.items():
                print(f"{k} Notes: {v}")
            print("---" * 10)


if __name__ == "__main__":

    preprocessor = PreProcess(model, device, dataset, IDF_weights=IDF_weight)
    emb_DB = preprocessor.get_emb_dict()
    names, DB_batch = preprocessor.dict_to_tensor(emb_DB)

    recommender = Perfume_Recommender(
        dataset=dataset,
        model=model,
        device=device,
        layer_weights=layer_weights,
        DB_batch=DB_batch,
        names=names,
        IDF_weights=IDF_weight,
    )

    # 커스텀 노트로 추천
    query_notes = {
        "Top": "rose",
        "Middle": "sunflower",
        "Base": "pear",
    }
    query_notes, top_scores, top_indices = recommender.recommend(query_notes=query_notes, n_recommendations=5)
    recommender.display_recommendations(query_notes, "Custom Query", top_scores, top_indices)

    print()

    # DB 인덱스로 추천
    target_notes, target_name, top_scores, top_indices = recommender.recommend_idx(query_idx="random", n_recommendations=5)
    recommender.display_recommendations(target_notes, target_name, top_scores, top_indices)
