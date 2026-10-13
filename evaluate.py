"""
향수 추천 모델 평가 스크립트
perfume_recommendation_v2.py 기반 모델의 성능을 다양한 지표로 평가합니다.
"""

from sentence_transformers import SentenceTransformer
import pandas as pd
import torch
from torch import nn
import numpy as np
from collections import defaultdict
from functions import get_IDF_weights
from perfume_recommendation_v2 import PreProcess, Perfume_Recommender

device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
print(f"USING DEVICE: {device}")

# ── 모델 및 데이터 로드 ──
model = SentenceTransformer("all-MiniLM-L6-v2").to(device)
dataset = pd.read_csv("fra_cleaned.csv", sep=";", encoding="unicode_escape", on_bad_lines="skip")
IDF_weight = get_IDF_weights(dataset)

layer_weights = torch.tensor([[0.8, 0.25, 0.1],
                               [0.25, 1., 0.35],
                               [0.1, 0.35, 1.2]], device=device)
layer_weights = layer_weights / torch.sum(layer_weights)

# ── 전처리 및 추천기 초기화 ──
preprocessor = PreProcess(model, device, dataset, IDF_weights=IDF_weight)
emb_DB = preprocessor.get_emb_dict()
names, DB_batch = preprocessor.dict_to_tensor(emb_DB)

recommender = Perfume_Recommender(
    dataset=dataset, model=model, device=device,
    layer_weights=layer_weights, DB_batch=DB_batch,
    names=names, IDF_weights=IDF_weight,
)

N = len(dataset)
K_VALUES = [1, 3, 5, 10]
MAX_K = max(K_VALUES)


def compute_all_scores():
    """모든 향수에 대해 전체 DB와의 유사도 점수를 계산합니다."""
    all_scores = []
    all_indices = []

    for query_idx in range(N):
        target_notes = {
            "Top": dataset.at[query_idx, "Top"],
            "Middle": dataset.at[query_idx, "Middle"],
            "Base": dataset.at[query_idx, "Base"],
        }
        weighted_emb = recommender.process_notes(target_notes)
        q_norm = nn.functional.normalize(weighted_emb, p=2, dim=1)
        db_norm = nn.functional.normalize(recommender.DB_batch, p=2, dim=2)
        similarity_map = torch.einsum("id, bjd -> bij", q_norm, db_norm)
        scores = (similarity_map * recommender.layer_weights).sum(dim=(1, 2))

        top_scores, top_indices = torch.topk(scores, k=min(MAX_K + 1, N))
        all_scores.append(top_scores.cpu())
        all_indices.append(top_indices.cpu())

        if (query_idx + 1) % 500 == 0:
            print(f"  [{query_idx + 1}/{N}] 점수 계산 완료")

    return all_scores, all_indices


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. Hit@K & MRR (자기 자신 검색)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def eval_hit_at_k_and_mrr(all_indices):
    hits = {k: 0 for k in K_VALUES}
    reciprocal_ranks = []

    for query_idx in range(N):
        target_name = dataset.at[query_idx, "Perfume"]
        top_idx = all_indices[query_idx].tolist()
        top_names = [names[i] for i in top_idx]

        # 자기 자신의 rank 찾기
        rank = None
        for r, name in enumerate(top_names):
            if name == target_name:
                rank = r + 1  # 1-based
                break

        if rank is not None:
            reciprocal_ranks.append(1.0 / rank)
            for k in K_VALUES:
                if rank <= k:
                    hits[k] += 1
        else:
            reciprocal_ranks.append(0.0)

    hit_at_k = {k: hits[k] / N for k in K_VALUES}
    mrr = np.mean(reciprocal_ranks)
    return hit_at_k, mrr


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. Brand Consistency (브랜드 일관성)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def eval_brand_consistency(all_indices, top_n=5):
    """추천된 top_n 향수 중 같은 브랜드 향수의 비율"""
    brand_hit_counts = []

    for query_idx in range(N):
        target_brand = dataset.at[query_idx, "Brand"]
        target_name = dataset.at[query_idx, "Perfume"]
        top_idx = all_indices[query_idx].tolist()
        top_names = [names[i] for i in top_idx]

        # 자기 자신 제거
        if target_name in top_names:
            top_names.remove(target_name)
        top_names = top_names[:top_n]

        same_brand = 0
        for rec_name in top_names:
            rec_brand_match = dataset.loc[dataset["Perfume"].str.strip() == rec_name, "Brand"]
            if len(rec_brand_match) > 0 and rec_brand_match.values[0] == target_brand:
                same_brand += 1

        brand_hit_counts.append(same_brand / top_n)

    return np.mean(brand_hit_counts)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. Note Overlap (노트 겹침도)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def eval_note_overlap(all_indices, top_n=5):
    """추천된 향수와 원본 향수의 노트 Jaccard 유사도"""
    overlaps = {layer: [] for layer in ["Top", "Middle", "Base"]}
    overall_overlaps = []

    for query_idx in range(N):
        target_name = dataset.at[query_idx, "Perfume"]
        target_notes = {}
        all_target_notes = set()
        for layer in ["Top", "Middle", "Base"]:
            notes = {n.strip() for n in dataset.at[query_idx, layer].split(",")}
            target_notes[layer] = notes
            all_target_notes.update(notes)

        top_idx = all_indices[query_idx].tolist()
        top_names = [names[i] for i in top_idx]
        if target_name in top_names:
            top_names.remove(target_name)
        top_names = top_names[:top_n]

        for rec_name in top_names:
            rec_row = dataset.loc[dataset["Perfume"].str.strip() == rec_name]
            if len(rec_row) == 0:
                continue
            rec_row = rec_row.iloc[0]

            all_rec_notes = set()
            for layer in ["Top", "Middle", "Base"]:
                rec_notes = {n.strip() for n in rec_row[layer].split(",")}
                all_rec_notes.update(rec_notes)
                # Jaccard per layer
                intersection = target_notes[layer] & rec_notes
                union = target_notes[layer] | rec_notes
                if len(union) > 0:
                    overlaps[layer].append(len(intersection) / len(union))

            # Overall Jaccard
            intersection = all_target_notes & all_rec_notes
            union = all_target_notes | all_rec_notes
            if len(union) > 0:
                overall_overlaps.append(len(intersection) / len(union))

    layer_means = {layer: np.mean(vals) for layer, vals in overlaps.items()}
    overall_mean = np.mean(overall_overlaps)
    return layer_means, overall_mean


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 4. Score Distribution (점수 분포)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def eval_score_distribution(all_scores):
    """Top-1 추천(자기 자신 제외) 유사도 점수의 분포를 분석합니다."""
    top1_scores = []
    self_scores = []

    for query_idx in range(N):
        scores = all_scores[query_idx].tolist()
        target_name = dataset.at[query_idx, "Perfume"]

        self_scores.append(scores[0])  # 보통 자기 자신이 top-1
        if len(scores) > 1:
            top1_scores.append(scores[1])  # 자기 자신 다음으로 높은 점수

    return {
        "self_score": {"mean": np.mean(self_scores), "std": np.std(self_scores),
                       "min": np.min(self_scores), "max": np.max(self_scores)},
        "top1_score": {"mean": np.mean(top1_scores), "std": np.std(top1_scores),
                       "min": np.min(top1_scores), "max": np.max(top1_scores)},
        "score_gap": {"mean": np.mean(np.array(self_scores) - np.array(top1_scores)),
                      "std": np.std(np.array(self_scores) - np.array(top1_scores))},
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 5. Intra-brand vs Inter-brand 유사도
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def eval_brand_similarity(all_scores, all_indices, sample_size=500):
    """같은 브랜드 내 향수 유사도 vs 다른 브랜드 향수 유사도"""
    intra_scores = []
    inter_scores = []

    sample_indices = np.random.choice(N, size=min(sample_size, N), replace=False)

    for query_idx in sample_indices:
        target_brand = dataset.at[query_idx, "Brand"]
        target_name = dataset.at[query_idx, "Perfume"]

        top_idx = all_indices[query_idx].tolist()
        scores = all_scores[query_idx].tolist()
        top_names = [names[i] for i in top_idx]

        for i, rec_name in enumerate(top_names):
            if rec_name == target_name:
                continue
            rec_brand_match = dataset.loc[dataset["Perfume"].str.strip() == rec_name, "Brand"]
            if len(rec_brand_match) == 0:
                continue
            rec_brand = rec_brand_match.values[0]

            if rec_brand == target_brand:
                intra_scores.append(scores[i])
            else:
                inter_scores.append(scores[i])

    return {
        "intra_brand": {"mean": np.mean(intra_scores) if intra_scores else 0,
                        "count": len(intra_scores)},
        "inter_brand": {"mean": np.mean(inter_scores) if inter_scores else 0,
                        "count": len(inter_scores)},
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 6. Gender Consistency (성별 일관성)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def eval_gender_consistency(all_indices, top_n=5):
    """추천된 향수가 같은 성별 카테고리인 비율"""
    gender_hits = []

    for query_idx in range(N):
        target_gender = dataset.at[query_idx, "Gender"]
        target_name = dataset.at[query_idx, "Perfume"]
        top_idx = all_indices[query_idx].tolist()
        top_names = [names[i] for i in top_idx]

        if target_name in top_names:
            top_names.remove(target_name)
        top_names = top_names[:top_n]

        same_gender = 0
        for rec_name in top_names:
            rec_row = dataset.loc[dataset["Perfume"].str.strip() == rec_name]
            if len(rec_row) > 0 and rec_row.iloc[0]["Gender"] == target_gender:
                same_gender += 1

        gender_hits.append(same_gender / top_n)

    return np.mean(gender_hits)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  평가 실행
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
if __name__ == "__main__":
    print(f"\n데이터셋 크기: {N}개 향수")
    print(f"평가 K values: {K_VALUES}")
    print("=" * 60)

    print("\n[1/6] 전체 유사도 점수 계산 중...")
    all_scores, all_indices = compute_all_scores()

    print("\n[2/6] Hit@K & MRR 계산 중...")
    hit_at_k, mrr = eval_hit_at_k_and_mrr(all_indices)

    print("\n[3/6] Brand Consistency 계산 중...")
    brand_consistency = eval_brand_consistency(all_indices)

    print("\n[4/6] Note Overlap 계산 중...")
    layer_overlap, overall_overlap = eval_note_overlap(all_indices)

    print("\n[5/6] Score Distribution 분석 중...")
    score_dist = eval_score_distribution(all_scores)

    print("\n[6/6] Brand Similarity & Gender Consistency 계산 중...")
    brand_sim = eval_brand_similarity(all_scores, all_indices)
    gender_consistency = eval_gender_consistency(all_indices)

    # ── 결과 출력 ──
    print("\n" + "=" * 60)
    print("              향수 추천 모델 평가 결과")
    print("=" * 60)

    print("\n[1] Self-Retrieval (자기 자신 검색 성능)")
    for k in K_VALUES:
        print(f"    Hit@{k:<3}: {hit_at_k[k]*100:6.2f}%")
    print(f"    MRR   : {mrr:.4f}")

    print(f"\n[2] Brand Consistency (Top-5 브랜드 일치율)")
    print(f"    평균  : {brand_consistency*100:.2f}%")

    print(f"\n[3] Note Overlap (Top-5 노트 Jaccard 유사도)")
    for layer, val in layer_overlap.items():
        print(f"    {layer:<7}: {val:.4f}")
    print(f"    Overall: {overall_overlap:.4f}")

    print(f"\n[4] Score Distribution (유사도 점수 분포)")
    print(f"    Self score   : mean={score_dist['self_score']['mean']:.4f}, std={score_dist['self_score']['std']:.4f}")
    print(f"    Top-1 score  : mean={score_dist['top1_score']['mean']:.4f}, std={score_dist['top1_score']['std']:.4f}")
    print(f"    Score gap    : mean={score_dist['score_gap']['mean']:.4f}, std={score_dist['score_gap']['std']:.4f}")

    print(f"\n[5] Intra vs Inter Brand Similarity")
    print(f"    같은 브랜드  : mean={brand_sim['intra_brand']['mean']:.4f} (n={brand_sim['intra_brand']['count']})")
    print(f"    다른 브랜드  : mean={brand_sim['inter_brand']['mean']:.4f} (n={brand_sim['inter_brand']['count']})")

    print(f"\n[6] Gender Consistency (Top-5 성별 일치율)")
    print(f"    평균  : {gender_consistency*100:.2f}%")

    print("\n" + "=" * 60)
