import numpy as np
import math
from itertools import combinations

def calculate_hit_and_ndcg(recommended_list, target_item):
    """
    단일 유저에 대한 Hit Ratio와 NDCG를 계산합니다.
    - recommended_list: 모델이 추천한 Top-K 아이템 리스트
    - target_item: 실제 Leave-One-Out 정답 아이템 (1개)
    """
    if target_item in recommended_list:
        hit = 1.0
        # 0-based index이므로 +2를 해줌 (1등이면 index 0 -> log2(2) = 1)
        rank = recommended_list.index(target_item)
        ndcg = 1.0 / math.log2(rank + 2)
    else:
        hit = 0.0
        ndcg = 0.0
        
    return hit, ndcg

def calculate_coverage(all_recommended_items_list, total_num_items):
    """
    전체 유저에게 추천된 아이템의 커버리지를 계산합니다.
    - all_recommended_items_list: 모든 유저의 Top-K 추천 리스트가 담긴 2차원 리스트
    - total_num_items: 데이터셋의 전체 아이템 개수
    """
    unique_recommended_items = set()
    for rec_list in all_recommended_items_list:
        unique_recommended_items.update(rec_list)
        
    coverage = len(unique_recommended_items) / total_num_items
    return coverage

def calculate_novelty(recommended_list, item_popularity_dict, total_interactions):
    """
    단일 유저 추천 리스트의 참신성(Novelty)을 계산합니다.
    - item_popularity_dict: Train 셋에서 각 아이템이 소비된 횟수 (dict)
    """
    novelty_score = 0.0
    for item in recommended_list:
        # 아이템의 인기도 (등장 확률)
        p_i = item_popularity_dict.get(item, 1) / total_interactions
        novelty_score += -math.log2(p_i)
        
    return novelty_score / len(recommended_list)

def calculate_serendipity(recommended_list, target_item, item_popularity_dict, total_interactions):
    """
    우연한 발견(Serendipity)을 계산합니다. (Hit를 한 아이템의 Novelty)
    맞추지 못했으면 0점, 맞췄다면 그 아이템이 비인기 상품일수록 높은 점수 부여.
    """
    if target_item in recommended_list:
        p_i = item_popularity_dict.get(target_item, 1) / total_interactions
        return -math.log2(p_i)
    else:
        return 0.0

def calculate_diversity_ild(recommended_list, item_embeddings):
    """
    목록 내 다양성(Intra-List Diversity)을 계산합니다.
    아이템 임베딩 간의 쌍(Pairwise) 코사인 거리를 평균냅니다.
    - item_embeddings: 모델이 학습한 전체 아이템의 벡터 (numpy array)
    """
    if len(recommended_list) < 2:
        return 0.0
    
    distance_sum = 0.0
    pairs = list(combinations(recommended_list, 2))
    
    for item_i, item_j in pairs:
        vec_i = item_embeddings[item_i]
        vec_j = item_embeddings[item_j]
        
        # 분모 0 방지 및 코사인 유사도 계산
        norm_i = np.linalg.norm(vec_i)
        norm_j = np.linalg.norm(vec_j)
        if norm_i == 0 or norm_j == 0:
            sim = 0
        else:
            sim = np.dot(vec_i, vec_j) / (norm_i * norm_j)
            
        distance = 1.0 - sim # 유사할수록 거리는 0, 다를수록 1
        distance_sum += distance
        
    # 모든 쌍의 거리에 대한 평균 반환
    ild = distance_sum / len(pairs)
    return ild