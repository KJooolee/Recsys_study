import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils.metrics import (
    calculate_hit_and_ndcg,
    calculate_coverage,
    calculate_novelty,
    calculate_serendipity,
    calculate_diversity_ild
)

def evaluate_model(model, test_loader, train_df, num_items, device, k=10, source='Dataset', model_name='Model'):
    """
    학습된 모델(BPR-MF, LightGCN, SASRec 모두 호환)을 Test 셋으로 평가하고,
    모든 지표를 계산하여 결과 그래프와 텍스트로 저장합니다.
    """
    model.eval()
    
    hits, ndcgs = [], []
    all_recommended_lists = []
    all_target_items = []
    
    item_counts = train_df['item_id'].value_counts().to_dict()
    total_interactions = len(train_df)
    
    # 모델이 SASRec인지 확인하는 플래그
    is_sasrec = (model_name == 'SASRec')
    
    print(f"\n[{model_name}] Test 셋 평가 및 지표 계산을 시작합니다...")
    with torch.no_grad():
        # 데이터로더가 뱉어내는 덩어리(batch)를 통째로 받음
        for batch in tqdm(test_loader, desc="Evaluating"):
            
            # 1. 모델 종류에 따른 데이터 Unpacking (분해)
            if is_sasrec:
                # SASRec은 4개의 변수(유저, 시퀀스, 정답, 오답)를 반환
                users, seqs, target_item, neg_items = batch
                seqs = seqs.to(device)
            else:
                # BPR 계열은 3개의 변수(유저, 정답, 오답)를 반환
                users, target_item, neg_items = batch
                
            users = users.to(device)
            target_item = target_item.to(device)
            neg_items = neg_items.to(device)
            batch_size, num_negs = neg_items.shape
            
            # 2. 모델 종류에 따른 추론(Prediction) 로직
            if is_sasrec:
                # SASRec: 시퀀스에서 맥락(log_feats)을 뽑은 뒤 점수 계산
                log_feats = model(seqs) 
                pos_scores = model.predict(log_feats, target_item)
                neg_scores = model.predict(log_feats, neg_items)
                
            else:
                # BPR-MF / LightGCN: 유저-아이템 직접 내적 연산
                pos_scores = model.predict(users, target_item)
                neg_scores = model.predict(users, neg_items)
                
            # 3. 랭킹 산출 (이후 과정은 모든 모델이 완벽하게 동일)
            all_scores = torch.cat([pos_scores.unsqueeze(1), neg_scores], dim=1)
            all_item_ids = torch.cat([target_item.unsqueeze(1), neg_items], dim=1)
            
            _, top_indices = torch.topk(all_scores, k=k, dim=1)
            
            # 배치별 추천 결과와 정답을 단 한 번만 CPU로 이동 (수만 번의 CPU-GPU 동기화 병목 제거)
            rec_lists = torch.gather(all_item_ids, 1, top_indices).cpu().numpy().tolist()
            targets_np = target_item.cpu().numpy().tolist()
            
            for i in range(batch_size):
                rec_list = rec_lists[i]
                target = targets_np[i]
                
                hit, ndcg = calculate_hit_and_ndcg(rec_list, target)
                hits.append(hit)
                ndcgs.append(ndcg)
                
                all_recommended_lists.append(rec_list)
                all_target_items.append(target)

    # 4. 전체 지표(Metrics) 집계
    print("\n세부 지표(Diversity, Novelty 등)를 연산 중입니다. 잠시만 기다려주세요...")
    
    hr_score = np.mean(hits)
    ndcg_score = np.mean(ndcgs)
    coverage_score = calculate_coverage(all_recommended_lists, num_items)
    
    novelties = [calculate_novelty(rec, item_counts, total_interactions) for rec in all_recommended_lists]
    novelty_score = np.mean(novelties)
    
    serendipities = [calculate_serendipity(rec, tgt, item_counts, total_interactions) for rec, tgt in zip(all_recommended_lists, all_target_items)]
    serendipity_score = np.mean(serendipities)
    
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
    
    # 5. 결과 저장 및 시각화
    save_evaluation_results(metrics, source, model_name)
    
    return metrics

def save_evaluation_results(metrics, source, model_name):
    # (이 함수는 기존과 100% 동일합니다. 변경할 필요 없습니다)
    save_dir = f'results/{source}/eval'
    os.makedirs(save_dir, exist_ok=True)
    
    txt_path = os.path.join(save_dir, f'{model_name}_metrics.txt')
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(f"=== {model_name} on {source} Dataset ===\n")
        for k, v in metrics.items():
            f.write(f"- {k}: {v:.4f}\n")
    print(f"평가 수치 리포트 저장 완료: {txt_path}")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    ratio_metrics = ['Hit@10', 'NDCG@10', 'Coverage', 'Diversity (ILD)']
    ratio_values = [metrics[m] for m in ratio_metrics]
    axes[0].bar(ratio_metrics, ratio_values, color=['#4C72B0', '#55A868', '#C44E52', '#8172B3'])
    axes[0].set_ylim(0, 1.0)
    axes[0].set_title('Ratio & Distance Metrics (0 to 1)', fontsize=14)
    for i, v in enumerate(ratio_values):
        axes[0].text(i, v + 0.02, f"{v:.4f}", ha='center', fontweight='bold')
        
    absolute_metrics = ['Novelty', 'Serendipity']
    absolute_values = [metrics[m] for m in absolute_metrics]
    axes[1].bar(absolute_metrics, absolute_values, color=['#CCB974', '#64B5CD'])
    axes[1].set_title('Absolute Score Metrics', fontsize=14)
    axes[1].set_ylim(0, max(absolute_values) * 1.2 if max(absolute_values) > 0 else 1)
    for i, v in enumerate(absolute_values):
        axes[1].text(i, v + 0.05, f"{v:.4f}", ha='center', fontweight='bold')
        
    plt.suptitle(f'Evaluation Results: {model_name} ({source})', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    img_path = os.path.join(save_dir, f'{model_name}_performance.png')
    plt.savefig(img_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"성능 그래프 이미지 저장 완료: {img_path}")