"""
Layer weight 최적화 스크립트
Hit@1 (self-retrieval)을 최대화하는 layer_weights를 찾습니다.
"""

from sentence_transformers import SentenceTransformer
import pandas as pd
import torch
from torch import nn
import numpy as np
from scipy.optimize import minimize
from functions import get_IDF_weights
from perfume_recommendation_v2 import PreProcess
import time

device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
print(f"USING DEVICE: {device}")

# ── 데이터 로드 ──
model = SentenceTransformer("all-MiniLM-L6-v2").to(device)
dataset = pd.read_csv("fra_cleaned.csv", sep=";", encoding="unicode_escape", on_bad_lines="skip")
IDF_weight = get_IDF_weights(dataset)
N = len(dataset)

# ── 임베딩 전처리 (한 번만) ──
print(f"데이터셋: {N}개 향수")
print("임베딩 전처리 중...")
preprocessor = PreProcess(model, device, dataset, IDF_weights=IDF_weight)
emb_DB = preprocessor.get_emb_dict()
names, DB_batch = preprocessor.dict_to_tensor(emb_DB)

# 모든 쿼리의 normalized embedding 미리 계산
print("쿼리 임베딩 정규화 중...")
all_q_norm = []
for query_idx in range(N):
    target_notes = {
        "Top": dataset.at[query_idx, "Top"],
        "Middle": dataset.at[query_idx, "Middle"],
        "Base": dataset.at[query_idx, "Base"],
    }
    weighted_emb = preprocessor.process_notes(target_notes)
    q_norm = nn.functional.normalize(weighted_emb, p=2, dim=1)
    all_q_norm.append(q_norm)
    if (query_idx + 1) % 5000 == 0:
        print(f"  [{query_idx + 1}/{N}]")

all_q_norm = torch.stack(all_q_norm)  # (N, 3, 384)
db_norm = nn.functional.normalize(DB_batch, p=2, dim=2)  # (N_db, 3, 384)
N_DB = db_norm.shape[0]

print(f"전처리 완료. all_q_norm: {all_q_norm.shape}, db_norm: {db_norm.shape}")

# name -> index 매핑 (DB 기준)
name_to_idx = {name: i for i, name in enumerate(names)}
# 각 쿼리의 정답 인덱스 (DB에서의 위치)
true_indices = []
valid_queries = []  # DB에 존재하는 쿼리만
for query_idx in range(N):
    target_name = dataset.at[query_idx, "Perfume"]
    idx = name_to_idx.get(target_name, -1)
    true_indices.append(idx)
    if idx >= 0:
        valid_queries.append(query_idx)
true_indices = torch.tensor(true_indices, device=device)
print(f"유효한 쿼리: {len(valid_queries)}/{N} (중복 제거된 DB: {N_DB})")


def params_to_weights(params):
    """6개 파라미터 -> 3x3 대칭 행렬 (softplus로 양수 보장)"""
    p = torch.tensor(params, dtype=torch.float32, device=device)
    p = nn.functional.softplus(p)  # 양수 보장
    W = torch.zeros(3, 3, device=device)
    W[0, 0] = p[0]
    W[0, 1] = W[1, 0] = p[1]
    W[0, 2] = W[2, 0] = p[2]
    W[1, 1] = p[3]
    W[1, 2] = W[2, 1] = p[4]
    W[2, 2] = p[5]
    W = W / W.sum()
    return W


def eval_hit_at_1(params, sample_indices=None):
    """주어진 파라미터로 Hit@1 계산 (음수 반환 = minimize용)"""
    weights = params_to_weights(params)

    if sample_indices is None:
        indices = range(N)
        n_eval = N
    else:
        indices = sample_indices
        n_eval = len(indices)

    hits = 0
    for qi in indices:
        true_idx = true_indices[qi].item()
        if true_idx < 0:
            continue
        # q: (3, 384), db: (N_DB, 3, 384)
        # sim_map[j, i, k] = cosine(q_layer_i, db_j_layer_k)
        q = all_q_norm[qi]  # (3, 384)
        sim_map = torch.einsum("id, bjd -> bij", q, db_norm)  # (N_DB, 3, 3)
        scores = (sim_map * weights).sum(dim=(1, 2))  # (N_DB,)
        top_idx = torch.argmax(scores).item()
        if top_idx == true_idx:
            hits += 1

    return hits / n_eval


def objective(params, sample_indices=None):
    """최소화 목적함수 (1 - Hit@1)"""
    hit1 = eval_hit_at_1(params, sample_indices)
    return -hit1  # 최대화 -> 최소화


# ── 최적화 ──
print("\n" + "=" * 60)
print("최적화 시작")
print("=" * 60)

# 현재 weights의 파라미터 역산 (초기값)
# 현재: [[0.8, 0.25, 0.1], [0.25, 1., 0.35], [0.1, 0.35, 1.2]]
# softplus 역함수: log(exp(x) - 1)
current_raw = [0.8, 0.25, 0.1, 1.0, 0.35, 1.2]
x0 = [np.log(np.exp(v) - 1) if v > 0 else 0.0 for v in current_raw]

# Phase 1: 샘플로 빠른 탐색
np.random.seed(42)
sample_size = 3000
sample_indices = np.random.choice(valid_queries, size=min(sample_size, len(valid_queries)), replace=False).tolist()

print(f"\n[Phase 1] {sample_size}개 샘플로 Nelder-Mead 최적화")
current_hit1 = eval_hit_at_1(x0, sample_indices)
print(f"현재 weights Hit@1 (sample): {current_hit1 * 100:.2f}%")

iteration_count = [0]
best_so_far = [current_hit1]

def callback(xk):
    iteration_count[0] += 1
    if iteration_count[0] % 5 == 0:
        hit1 = eval_hit_at_1(xk, sample_indices)
        if hit1 > best_so_far[0]:
            best_so_far[0] = hit1
        print(f"  Iter {iteration_count[0]:3d}: Hit@1 = {hit1 * 100:.2f}% (best: {best_so_far[0] * 100:.2f}%)")

start = time.time()
result = minimize(
    objective,
    x0,
    args=(sample_indices,),
    method="Nelder-Mead",
    callback=callback,
    options={"maxiter": 200, "xatol": 0.01, "fatol": 0.001, "adaptive": True},
)
elapsed = time.time() - start
print(f"Phase 1 완료: {elapsed:.1f}초")

# Phase 1 결과
best_params_phase1 = result.x
optimized_weights_phase1 = params_to_weights(best_params_phase1)
hit1_phase1 = eval_hit_at_1(best_params_phase1, sample_indices)
print(f"Phase 1 최적 Hit@1 (sample): {hit1_phase1 * 100:.2f}%")
print(f"Phase 1 최적 weights:\n{optimized_weights_phase1 * optimized_weights_phase1.sum()}")

# Phase 2: 더 큰 샘플로 fine-tuning
sample_size_2 = 6000
sample_indices_2 = np.random.choice(valid_queries, size=min(sample_size_2, len(valid_queries)), replace=False).tolist()

print(f"\n[Phase 2] {sample_size_2}개 샘플로 fine-tuning")
iteration_count[0] = 0
best_so_far[0] = eval_hit_at_1(best_params_phase1, sample_indices_2)
print(f"Phase 1 결과의 Hit@1 (larger sample): {best_so_far[0] * 100:.2f}%")

start = time.time()
result2 = minimize(
    objective,
    best_params_phase1,
    args=(sample_indices_2,),
    method="Nelder-Mead",
    callback=callback,
    options={"maxiter": 150, "xatol": 0.005, "fatol": 0.0005, "adaptive": True},
)
elapsed = time.time() - start
print(f"Phase 2 완료: {elapsed:.1f}초")

best_params = result2.x
optimized_weights = params_to_weights(best_params)

# ── 전체 데이터셋으로 최종 평가 ──
print(f"\n[최종 평가] 전체 {N}개 향수로 검증 중...")

# 원래 weights
original_params = x0
original_hit1 = eval_hit_at_1(original_params)
print(f"원래 weights Hit@1: {original_hit1 * 100:.2f}%")

# 최적화된 weights
optimized_hit1 = eval_hit_at_1(best_params)
print(f"최적화 weights Hit@1: {optimized_hit1 * 100:.2f}%")

# ── 결과 출력 ──
print("\n" + "=" * 60)
print("              최적화 결과")
print("=" * 60)

# 정규화 전 원래 스케일로 복원
raw_weights = optimized_weights * optimized_weights.sum()  # 정규화 해제 (sum=1이므로 그대로)
# 보기 좋게 총합 기준으로 스케일링
scale = 3.35 / optimized_weights.sum()  # 원래 총합과 비슷하게
display_weights = optimized_weights * optimized_weights.sum() * scale

print(f"\n원래 layer_weights:")
original_W = torch.tensor([[0.8, 0.25, 0.1],
                            [0.25, 1., 0.35],
                            [0.1, 0.35, 1.2]])
print(original_W)
print(f"Hit@1: {original_hit1 * 100:.2f}%")

print(f"\n최적화된 layer_weights (정규화 전):")
# 총합을 원래와 비슷하게 스케일링
total_original = original_W.sum().item()
scaled_weights = optimized_weights * total_original
print(torch.round(scaled_weights * 1000) / 1000)
print(f"Hit@1: {optimized_hit1 * 100:.2f}%")

print(f"\n개선폭: {(optimized_hit1 - original_hit1) * 100:+.2f}%p")

print(f"\n코드에 적용할 값:")
w = optimized_weights * total_original
print(f"layer_weights = torch.tensor([[{w[0,0]:.4f}, {w[0,1]:.4f}, {w[0,2]:.4f}],")
print(f"                               [{w[1,0]:.4f}, {w[1,1]:.4f}, {w[1,2]:.4f}],")
print(f"                               [{w[2,0]:.4f}, {w[2,1]:.4f}, {w[2,2]:.4f}]], device=device)")

# MRR도 계산
print(f"\n[추가] MRR 비교")
def eval_mrr(params):
    weights = params_to_weights(params)
    reciprocal_ranks = []
    for qi in valid_queries:
        true_idx = true_indices[qi].item()
        if true_idx < 0:
            continue
        q = all_q_norm[qi]
        sim_map = torch.einsum("id, bjd -> bij", q, db_norm)  # (N_DB, 3, 3)
        scores = (sim_map * weights).sum(dim=(1, 2))
        sorted_indices = torch.argsort(scores, descending=True)
        rank = (sorted_indices == true_idx).nonzero(as_tuple=True)[0].item() + 1
        reciprocal_ranks.append(1.0 / rank)
    return np.mean(reciprocal_ranks)

original_mrr = eval_mrr(original_params)
optimized_mrr = eval_mrr(best_params)
print(f"원래 MRR:    {original_mrr:.4f}")
print(f"최적화 MRR:  {optimized_mrr:.4f}")
print(f"개선폭:      {(optimized_mrr - original_mrr):+.4f}")

print("\n" + "=" * 60)
