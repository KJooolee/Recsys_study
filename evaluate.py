import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

# 우리가 만든 평가 지표 함수들 불러오기
from utils.metrics import (
    calculate_hit_and_ndcg,
    calculate_coverage,
    calculate_novelty,
    calculate_serendipity,
    calculate_diversity_ild
)

def evaluate_model(model, test_loader, train_df, num_items, device, k=10, source='Dataset', model_name='Model'):
    """
    학습된 모델을 Test 셋으로 평가하고, 모든 지표(Metrics)를 계산한 뒤
    결과를 그래프와 텍스트 파일로 저장합니다.
    """
    model.eval()
    
    # 지표 누적을 위한 리스트
    hits, ndcgs = [], []
    all_recommended_lists = []
    all_target_items = []
    
    # 1. 인기도(Popularity) 딕셔너리 생성 (Novelty, Serendipity 계산용)
    item_counts = train_df['item_id'].value_counts().to_dict()
    total_interactions = len(train_df)
    
    print(f"\n🔍 [{model_name}] Test 셋 평가 및 지표 계산을 시작합니다...")
    with torch.no_grad():
        for users, target_item, neg_items in tqdm(test_loader, desc="Evaluating"):
            users, target_item, neg_items = users.to(device), target_item.to(device), neg_items.to(device)
            batch_size, num_negs = neg_items.shape
            
            # 예측 점수 계산
            pos_scores = model.predict(users, target_item)
            
            users_expanded = users.unsqueeze(1).expand(-1, num_negs).reshape(-1)
            neg_items_flat = neg_items.reshape(-1)
            neg_scores = model.predict(users_expanded, neg_items_flat)
            neg_scores = neg_scores.view(batch_size, num_negs)
            
            # [batch_size, 100] 형태의 점수 행렬 (0번 인덱스가 정답)
            all_scores = torch.cat([pos_scores.unsqueeze(1), neg_scores], dim=1)
            
            # 실제 아이템 ID [batch_size, 100]
            all_item_ids = torch.cat([target_item.unsqueeze(1), neg_items], dim=1)
            
            # Top-K 추출 (점수가 가장 높은 K개의 인덱스)
            _, top_indices = torch.topk(all_scores, k=k, dim=1)
            
            # 배치 내 각 유저별로 지표 계산
            for i in range(batch_size):
                rec_list = all_item_ids[i, top_indices[i]].cpu().numpy().tolist()
                target = target_item[i].item()
                
                # Hit & NDCG 계산
                hit, ndcg = calculate_hit_and_ndcg(rec_list, target)
                hits.append(hit)
                ndcgs.append(ndcg)
                
                all_recommended_lists.append(rec_list)
                all_target_items.append(target)

    # 2. 전체 지표(Metrics) 집계
    print("\n📊 세부 지표(Diversity, Novelty 등)를 연산 중입니다. 잠시만 기다려주세요...")
    
    hr_score = np.mean(hits)
    ndcg_score = np.mean(ndcgs)
    coverage_score = calculate_coverage(all_recommended_lists, num_items)
    
    # Novelty & Serendipity (전체 평균)
    novelties = [calculate_novelty(rec, item_counts, total_interactions) for rec in all_recommended_lists]
    novelty_score = np.mean(novelties)
    
    serendipities = [calculate_serendipity(rec, tgt, item_counts, total_interactions) 
                     for rec, tgt in zip(all_recommended_lists, all_target_items)]
    serendipity_score = np.mean(serendipities)
    
    # Diversity (ILD) - 모델의 전체 아이템 임베딩을 가져와 계산
    item_embeddings = model.get_all_item_embeddings()
    diversities = [calculate_diversity_ild(rec, item_embeddings) for rec in all_recommended_lists]
    diversity_score = np.mean(diversities)
    
    metrics = {
        f'Hit@{k}': hr_score,
        f'NDCG@{k}': ndcg_score,
        'Coverage': coverage_score,
        'Diversity (ILD)': diversity_score,
        'Novelty': novelty_score,
        'Serendipity': serendipity_score
    }
    
    # 3. 결과 저장 및 시각화
    save_evaluation_results(metrics, source, model_name)
    
    return metrics

def save_evaluation_results(metrics, source, model_name):
    """지표를 텍스트로 저장하고 막대 그래프로 시각화합니다."""
    save_dir = f'results/{source}/eval'
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. 텍스트 파일 리포트 저장
    txt_path = os.path.join(save_dir, f'{model_name}_metrics.txt')
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(f"=== {model_name} on {source} Dataset ===\n")
        for k, v in metrics.items():
            f.write(f"- {k}: {v:.4f}\n")
    print(f"✅ 평가 수치 리포트 저장 완료: {txt_path}")
    
    # 2. 시각화 그래프 저장 (비율 지표와 절대 수치 지표를 분리하여 가독성 확보)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # A. 0~1 사이의 비율/거리 지표 (Hit, NDCG, Coverage, Diversity)
    ratio_metrics = ['Hit@10', 'NDCG@10', 'Coverage', 'Diversity (ILD)']
    ratio_values = [metrics[m] for m in ratio_metrics]
    
    axes[0].bar(ratio_metrics, ratio_values, color=['#4C72B0', '#55A868', '#C44E52', '#8172B3'])
    axes[0].set_ylim(0, 1.0)
    axes[0].set_title('Ratio & Distance Metrics (0 to 1)', fontsize=14)
    for i, v in enumerate(ratio_values):
        axes[0].text(i, v + 0.02, f"{v:.4f}", ha='center', fontweight='bold')
        
    # B. 절대 수치 지표 (Novelty, Serendipity)
    absolute_metrics = ['Novelty', 'Serendipity']
    absolute_values = [metrics[m] for m in absolute_metrics]
    
    axes[1].bar(absolute_metrics, absolute_values, color=['#CCB974', '#64B5CD'])
    axes[1].set_title('Absolute Score Metrics', fontsize=14)
    # y축 최대값을 값의 1.2배로 설정하여 글씨가 잘리지 않게 조정
    axes[1].set_ylim(0, max(absolute_values) * 1.2 if max(absolute_values) > 0 else 1)
    for i, v in enumerate(absolute_values):
        axes[1].text(i, v + 0.05, f"{v:.4f}", ha='center', fontweight='bold')
        
    plt.suptitle(f'Evaluation Results: {model_name} ({source})', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    img_path = os.path.join(save_dir, f'{model_name}_performance.png')
    plt.savefig(img_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 성능 그래프 이미지 저장 완료: {img_path}")