import torch
import torch.nn.functional as F
from tqdm import tqdm
from utils.early_stopping import EarlyStopping

class BPRTrainer:
    """
    BPR Loss 기반의 모델(MF, LightGCN 등)을 학습하고 검증하는 공통 엔진
    """
    def __init__(self, model, optimizer, device, epochs=50, save_path='best_model.pt'):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device
        self.epochs = epochs
        self.save_path = save_path
        self.early_stopping = EarlyStopping(patience=5, verbose=True, path=save_path, mode='max')

    def train_and_evaluate(self, train_loader, val_loader):
        for epoch in range(self.epochs):
            # == 1. Train Phase ==
            self.model.train()
            total_loss = 0.0
            
            loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.epochs} [Train]")
            for users, pos_items, neg_items in loop:
                users, pos_items, neg_items = users.to(self.device), pos_items.to(self.device), neg_items.to(self.device)
                
                self.optimizer.zero_grad()
                
                # 모델이 무엇이든 forward 결과만 받아서 처리 (다형성)
                pos_scores, neg_scores = self.model(users, pos_items, neg_items)
                
                # BPR Loss 계산
                loss = -F.logsigmoid(pos_scores - neg_scores).mean()
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                loop.set_postfix(loss=(total_loss/len(train_loader)))
                
            # == 2. Validation Phase ==
            val_hit_ratio = self.evaluate_hit_at_10(val_loader)
            print(f"Epoch {epoch+1} | Train Loss: {total_loss/len(train_loader):.4f} | Val Hit@10: {val_hit_ratio:.4f}")
            
            # == 3. Early Stopping ==
            self.early_stopping(val_hit_ratio, self.model)
            if self.early_stopping.early_stop:
                print("🚀 조기 종료(Early Stopping) 발동! 학습을 멈춥니다.")
                break
                
        return self.model

    def evaluate_hit_at_10(self, val_loader):
        """검증용 약식 평가 로직 (Hit@10)"""
        self.model.eval()
        val_hits = 0
        with torch.no_grad():
            for users, target_item, neg_items in tqdm(val_loader, desc="[Validation]"):
                users, target_item, neg_items = users.to(self.device), target_item.to(self.device), neg_items.to(self.device)
                
                batch_size, num_negs = neg_items.shape
                
                pos_scores = self.model.predict(users, target_item) 
                
                users_expanded = users.unsqueeze(1).expand(-1, num_negs).reshape(-1)
                neg_items_flat = neg_items.reshape(-1)
                
                neg_scores = self.model.predict(users_expanded, neg_items_flat)
                neg_scores = neg_scores.view(batch_size, num_negs)
                
                all_scores = torch.cat([pos_scores.unsqueeze(1), neg_scores], dim=1) 
                _, top_indices = torch.topk(all_scores, k=10, dim=1)
                
                hits = (top_indices == 0).sum().item()
                val_hits += hits
                
        return val_hits / len(val_loader.dataset)