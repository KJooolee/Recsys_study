# trainers/seq_trainer.py 파일 전체 교체

import torch
import torch.nn as nn
from tqdm import tqdm
from utils.early_stopping import EarlyStopping
from utils.plotter import plot_training_history

class SeqTrainer:
    def __init__(self, model, optimizer, device, epochs=50, save_path='best_seq_model.pt', eval_interval=2):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device
        self.epochs = epochs
        self.eval_interval = eval_interval
        # 추천 시스템에서는 일반적으로 Validation Ranking 지표(예: Hit@10)로 조기 종료를 수행합니다.
        self.early_stopping = EarlyStopping(patience=3, verbose=True, path=save_path, mode='max')
        
        self.bce_criterion = nn.BCEWithLogitsLoss(reduction='none')

        self.train_loss_history = []
        self.val_loss_history = []
        self.val_hr_history = []
        self.val_epochs_history = []
        self.metric_name = 'Val Hit@10'

    def train_and_evaluate(self, train_loader, val_loader):
        print("\n[SASRec] 학습을 시작합니다...")
        val_hr = 0.0
        val_loss = 0.0
        for epoch in range(self.epochs):
            # --- Train Phase ---
            self.model.train()
            total_loss = 0.0
            
            loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.epochs} [Train]")
            for users, seqs, pos, neg in loop:
                seqs, pos, neg = seqs.to(self.device), pos.to(self.device), neg.to(self.device)
                
                self.optimizer.zero_grad()
                
                log_feats = self.model(seqs) 
                
                pos_embs = self.model.item_emb(pos)
                neg_embs = self.model.item_emb(neg)
                
                pos_logits = (log_feats * pos_embs).sum(dim=-1)
                neg_logits = (log_feats * neg_embs).sum(dim=-1)
                
                mask = (pos != 0).float()
                
                pos_labels = torch.ones_like(pos_logits)
                neg_labels = torch.zeros_like(neg_logits)
                
                loss_pos = self.bce_criterion(pos_logits, pos_labels)
                loss_neg = self.bce_criterion(neg_logits, neg_labels)
                
                loss = (loss_pos * mask).sum() + (loss_neg * mask).sum()
                loss = loss / (mask.sum() + 1e-9) # NaN 방지를 위한 0 할당 방어코드
                
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                loop.set_postfix(loss=(total_loss/len(train_loader)))
            
            avg_train_loss = total_loss / len(train_loader)
            
            if (epoch + 1) % self.eval_interval == 0 or epoch == self.epochs - 1:
                # --- Validation Phase ---
                self.model.eval()
                val_hits = 0
                val_total_loss = 0.0
                with torch.no_grad():
                    for users, seqs, target_item, neg_items in tqdm(val_loader, desc="[Validation]", leave=False):
                        seqs, target_item, neg_items = seqs.to(self.device), target_item.to(self.device), neg_items.to(self.device)
                        batch_size, num_negs = neg_items.shape
                        
                        log_feats = self.model(seqs)
                        pos_scores = self.model.predict(log_feats, target_item)
                        neg_scores = self.model.predict(log_feats, neg_items)
                        
                        pos_labels = torch.ones_like(pos_scores)
                        neg_labels = torch.zeros_like(neg_scores)
                        loss_pos = nn.functional.binary_cross_entropy_with_logits(pos_scores, pos_labels)
                        loss_neg = nn.functional.binary_cross_entropy_with_logits(neg_scores, neg_labels)
                        val_total_loss += (loss_pos.item() + loss_neg.item())
                        
                        all_scores = torch.cat([pos_scores.unsqueeze(1), neg_scores], dim=1)
                        _, top_indices = torch.topk(all_scores, k=10, dim=1)
                        
                        hits = (top_indices == 0).sum().item()
                        val_hits += hits
                
                val_hr = val_hits / len(val_loader.dataset)
                val_loss = val_total_loss / len(val_loader)
                
                self.val_loss_history.append(val_loss)
                self.val_hr_history.append(val_hr)
                self.val_epochs_history.append(epoch + 1)
                
                print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.6f} | Val Loss: {val_loss:.6f} | {self.metric_name}: {val_hr:.4f}")
                
                # Early Stopping 체크 (검증셋 Hit@10 기준으로 진행)
                self.early_stopping(val_hr, self.model)
                if self.early_stopping.early_stop:
                    self.train_loss_history.append(avg_train_loss)
                    print(f"Early Stopping, {epoch+1} epoch에서 학습을 멈춥니다.")
                    break
            else:
                print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.6f} | (Validation 생략)")

            self.train_loss_history.append(avg_train_loss)
                
        print("\n학습 히스토리 그래프 생성 중...")
        plot_training_history(
            self.train_loss_history, 
            self.val_loss_history,
            self.val_hr_history, 
            self.val_epochs_history,
            self.metric_name, 
            self.early_stopping.path
        )

        self.model.load_state_dict(torch.load(self.early_stopping.path))
        return self.model