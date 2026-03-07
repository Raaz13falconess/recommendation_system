import torch
import numpy as np

def precision_at_k(recommended, relevant, k):
    recommended = recommended[:k]
    hits = len(set(recommended) & set(relevant))
    return hits / k

def recall_at_k(recommended, relevant, k):
    recommended = recommended[:k]
    hits = len(set(recommended) & set(relevant))
    return hits / len(relevant)

def ndcg_at_k(recommended, relevant, k):
    recommended = recommended[:k]

    dcg = 0.0
    for i, item in enumerate(recommended):
        if item in relevant:
            dcg += 1 / np.log2(i+2)
    
    ideal_hits = min(len(relevant), k)
    idcg = sum(1 / np.log2(i+2) for i in range(ideal_hits))

    return dcg / idcg if idcg > 0 else 0

