import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from utils.early_stopping import EarlyStopping

class SeqTrainer:
    """SASRec 등 시퀀셜 모델을 학습하기 위한 전용 Trainer"""
    def __init__(self, model, optimizer, device, epochs=50, save_path='best_seq_model.pt'):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device
        self.epochs = epochs
        self.early_stopping = EarlyStopping(patience=5, verbose=True, path=save_path, mode='max')
        
        # 시퀀스 예측에 주로 쓰이는 이진 교차 엔트로피 손실 함수
        self.bce_criterion = nn.BCEWithLogitsLoss(reduction='none')

    def train_and_evaluate(self, train_loader, val_loader):
        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0.0
            
            loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.epochs} [Train]")
            for users, seqs, pos, neg in loop:
                seqs, pos, neg = seqs.to(self.device), pos.to(self.device), neg.to(self.device)
                
                self.optimizer.zero_grad()
                
                # 모델에 시퀀스 입력 -> 은닉 상태(Hidden States) 출력
                log_feats = self.model(seqs) 
                
                # 타겟 아이템 임베딩 가져오기
                pos_embs = self.model.item_emb(pos)
                neg_embs = self.model.item_emb(neg)
                
                # 예측 점수 산출
                pos_logits = (log_feats * pos_embs).sum(dim=-1)
                neg_logits = (log_feats * neg_embs).sum(dim=-1)
                
                # 🚨 핵심: 패딩(0)인 부분은 Loss 계산에서 제외하기 위한 마스크 생성
                mask = (pos != 0).float()
                
                pos_labels = torch.ones_like(pos_logits)
                neg_labels = torch.zeros_like(neg_logits)
                
                loss_pos = self.bce_criterion(pos_logits, pos_labels)
                loss_neg = self.bce_criterion(neg_logits, neg_labels)
                
                # 마스크를 곱해서 실제 의미 있는 스텝의 Loss만 평균냄
                loss = (loss_pos * mask).sum() + (loss_neg * mask).sum()
                loss = loss / mask.sum() 
                
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                loop.set_postfix(loss=(total_loss/len(train_loader)))
                
            # Validation
            val_hit_ratio = self.evaluate_hit_at_10(val_loader)
            print(f"Epoch {epoch+1} | Train Loss: {total_loss/len(train_loader):.4f} | Val Hit@10: {val_hit_ratio:.4f}")
            
            self.early_stopping(val_hit_ratio, self.model)
            if self.early_stopping.early_stop:
                print("🚀 조기 종료(Early Stopping) 발동! 학습을 멈춥니다.")
                break
                
        return self.model

    def evaluate_hit_at_10(self, val_loader):
        """[검증용] Hit@10을 빠르게 계산합니다."""
        self.model.eval()
        val_hits = 0
        with torch.no_grad():
            for users, seqs, target_item, neg_items in tqdm(val_loader, desc="[Validation]"):
                seqs, target_item, neg_items = seqs.to(self.device), target_item.to(self.device), neg_items.to(self.device)
                batch_size, num_negs = neg_items.shape
                
                # 입력 시퀀스를 통해 가장 마지막 시점의 문맥을 파악
                log_feats = self.model(seqs)
                
                # 정답 점수 예측
                pos_scores = self.model.predict(log_feats, target_item)
                
                # 오답(네거티브) 점수 예측
                neg_items_flat = neg_items.reshape(-1)
                # 예측 대상 아이템 개수만큼 맥락 벡터 복제
                log_feats_expanded = log_feats.unsqueeze(1).expand(-1, num_negs, -1, -1).reshape(batch_size * num_negs, log_feats.size(1), -1)
                
                neg_scores = self.model.predict(log_feats_expanded, neg_items_flat)
                neg_scores = neg_scores.view(batch_size, num_negs)
                
                # 랭킹 산출
                all_scores = torch.cat([pos_scores.unsqueeze(1), neg_scores], dim=1)
                _, top_indices = torch.topk(all_scores, k=10, dim=1)
                
                hits = (top_indices == 0).sum().item()
                val_hits += hits
                
        return val_hits / len(val_loader.dataset)