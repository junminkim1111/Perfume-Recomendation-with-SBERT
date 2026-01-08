from sentence_transformers import SentenceTransformer
from sentence_transformers import util
import time
import pandas as pd
import torch
import random
from functions import get_IDF_weights


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"USING DEVICE: {device} / {torch.cuda.get_device_name() if torch.cuda.is_available() else None}")

# Data provided by FRAGRANTICA
# https://www.kaggle.com/datasets/olgagmiufana1/fragrantica-com-fragrance-dataset
dataset = pd.read_csv("fra_cleaned.csv",sep = ";" , encoding= 'unicode_escape', on_bad_lines='skip')

N = len(dataset)
note_types = ['Top', 'Middle', 'Base']

# 1. Load a pretrained Sentence Transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

layer_weights = torch.tensor([[0.8, 0.25, 0.1],
                            [0.25, 1., 0.35], 
                            [0.1, 0.35, 1.2]], device=device)

layer_weights = layer_weights / torch.sum(layer_weights)

class Perfume_Recommender:
    def __init__(self, dataset, model, device, layer_weights, ignore_IDF = True, IDF_weight = None):
        self.dataset = dataset
        self.model = model
        self.device = device
        self.layer_weights = layer_weights
        self.IDF_weight = IDF_weight if not ignore_IDF else {note: 1. for note in IDF_weight.keys()}
        self.note_embedding_cache = {}

    def timer(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            print(f"Duration: {end_time - start_time:.4f} seconds")
            return result
        return wrapper

    def get_note_embedding(self, note):
        '''
        Get the embedding for a given note.
        Caches the embedding to avoid redundant computations.
        '''

        if note not in self.note_embedding_cache:
            self.note_embedding_cache[note] = self.model.encode(
                note, convert_to_tensor=True, device=self.device
            )
        return self.note_embedding_cache[note]

    def process_notes(self, note_dict):
        '''
        Process notes by computing weighted average embeddings using IDF weights.
        1. multiply each note embedding by its normalised IDF weight
        2. sum the weighted embeddings to get a single embedding per note type
        3. stack the embeddings for all note types and return
        '''
        processed_notes = {}
        for k , v in note_dict.items():
            temp = [n.strip() for n in v.split(",")] # Remove extra spaces
            weight = torch.tensor([self.IDF_weight[w] for w in temp]).unsqueeze(1) # Get IDF weights
            weight = weight / torch.sum(weight)  # Normalize weights
            weight = weight.to(self.device) # Move weight to device(GPU/CPU)
            embeddings = torch.stack([self.get_note_embedding(n) for n in temp])
            processed_notes[k] = torch.sum(embeddings * weight, dim=0) # Compute weighted average embeddings
        
        weighted_embeddings = torch.vstack(list(processed_notes.values()))
        return weighted_embeddings

    def get_cosine_similarity(self, emb1, emb2, layer_weights):
        cosine_scores = util.cos_sim(emb1, emb2)
        score = torch.sum(cosine_scores * layer_weights).item() / torch.sum(layer_weights).item()
        return score
    
    @timer
    def recommend(self, target_idx = "random", n_recommendations=5):
        if target_idx == "random":
            target_idx = random.randint(0, len(self.dataset) - 1)
        target = {"Top": self.dataset.at[target_idx, "Top"],
                  "Middle": self.dataset.at[target_idx, "Middle"],
                  "Base": self.dataset.at[target_idx, "Base"]}
        weighted_embeddings_original = self.process_notes(target)
        sim_list = {}
        for iter in range(len(self.dataset)):
            compare_name = self.dataset.at[iter, "Perfume"]
            compare_notes = {"Top": self.dataset.at[iter, "Top"],
                             "Middle": self.dataset.at[iter, "Middle"],
                             "Base": self.dataset.at[iter, "Base"]}
            weighted_embeddings_compare = self.process_notes(compare_notes)
            score = self.get_cosine_similarity(weighted_embeddings_original, weighted_embeddings_compare, self.layer_weights)
            sim_list[compare_name] = score
        top_similars = sorted(sim_list, key=lambda k: sim_list[k], reverse=True)[:n_recommendations]
        return target, top_similars
    
    def display_recommendations(self, target, top_similars, note_comparison=True):

        target_name = self.dataset.loc[self.dataset['Perfume'].str.strip() == top_similars[0], 'Perfume'].item()
        print("***" * 10, "results", "***" * 10)
        print(f"Ignore IDF weight: ", "True" if all(v == 1.0 for v in self.IDF_weight.values()) else "False")
        print(f">>Original Perfume: {target_name}\n>>most similar perfume: {top_similars[0]}")
        print("successfully found top 5 similar perfumes" if target_name == top_similars[0] else "failed to find the original perfume")
        print("---" * 10)
        print(">>Top 5 similar perfumes:")
        for i, perfume in enumerate(top_similars):
            print(f"NO.{i+1} {perfume}")
        if note_comparison:
            print("---" * 10)
            print(f">>Most Similar Perfume except Original Perfume: {top_similars[1]}")
            print(f"Top Notes: {dataset.loc[dataset['Perfume'].str.strip() == top_similars[1], 'Top'].item()}")
            print(f"Middle Notes: {dataset.loc[dataset['Perfume'].str.strip() == top_similars[1], 'Middle'].item()}")
            print(f"Base Notes: {dataset.loc[dataset['Perfume'].str.strip() == top_similars[1], 'Base'].item()}")
            print("---" * 10)
            print(">>Original Notes:")
            for k, v in target.items():
                print(f"{k} Notes: {v}")
            print("---" * 10)


if __name__ == "__main__":

    random_num = random.randint(0, len(dataset) - 1)

    # ignore_IDF = False
    recommender1 = Perfume_Recommender(
        dataset = dataset,
        model = model,
        device = device,
        layer_weights = layer_weights,
        ignore_IDF = False,
        IDF_weight = get_IDF_weights(dataset, data_size = len(dataset)),
        )
    
    target, top_similars = recommender1.recommend(target_idx = random_num, n_recommendations=5)
    recommender1.display_recommendations(target, top_similars)

    # ignore_IDF = True
    recommender2 = Perfume_Recommender(
        dataset = dataset,
        model = model,
        device = device,
        layer_weights = layer_weights,
        ignore_IDF = True,
        IDF_weight = get_IDF_weights(dataset, data_size = len(dataset)),
        )
    target, top_similars = recommender2.recommend(target_idx = random_num, n_recommendations=5)
    recommender2.display_recommendations(target, top_similars)