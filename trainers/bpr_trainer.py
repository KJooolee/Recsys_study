# trainers/bpr_trainer.py 파일 전체 교체

import torch
from tqdm import tqdm
from utils.early_stopping import EarlyStopping
from utils.plotter import plot_training_history

class BPRTrainer:
    def __init__(self, model, optimizer, device, epochs=50, save_path='best_model.pt', eval_interval=2):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device
        self.epochs = epochs
        self.eval_interval = eval_interval
        # 추천 시스템에서는 일반적으로 Validation Ranking 지표(예: Hit@10)로 조기 종료를 수행합니다.
        self.early_stopping = EarlyStopping(patience=3, verbose=True, path=save_path, mode='max')
        
        self.train_loss_history = []
        self.val_loss_history = []
        self.val_hr_history = []
        self.val_epochs_history = []
        self.metric_name = 'Val Hit@10'

    def train_and_evaluate(self, train_loader, val_loader):
        print("\n학습을 시작합니다...")
        
        val_hr = 0.0
        val_loss = 0.0
        for epoch in range(self.epochs):
            # --- Train Phase ---
            self.model.train()
            total_loss = 0.0
            
            # 고속화 샘플러가 정의된 경우 로거의 병목(collate_fn)을 우회
            if hasattr(train_loader.dataset, "fast_sample"):
                batches = train_loader.dataset.fast_sample(train_loader.batch_size)
            else:
                batches = train_loader
                
            loop = tqdm(batches, desc=f"Epoch {epoch+1}/{self.epochs} [Train]")
            for users, pos_items, neg_items in loop:
                users, pos_items, neg_items = users.to(self.device), pos_items.to(self.device), neg_items.to(self.device)
                
                self.optimizer.zero_grad()
                
                output = self.model(users, pos_items, neg_items)
                
                if len(output) == 3:
                    pos_scores, neg_scores, reg_loss = output
                    loss = -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-9))
                    loss = loss + 1e-4 * reg_loss # 수동 Weight Decay
                else:
                    pos_scores, neg_scores = output
                    loss = -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-9))
                
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                loop.set_postfix(loss=(total_loss / (loop.n + 1)))
            
            avg_train_loss = total_loss / len(batches)
            
            if (epoch + 1) % self.eval_interval == 0 or epoch == self.epochs - 1:
                # --- Validation Phase ---
                self.model.eval()
                val_hits = 0.0
                val_total_loss = 0.0
                with torch.no_grad():
                    # 간단 검증을 위해 tqdm desc만 다르게
                    val_loop = tqdm(val_loader, desc=f"Epoch {epoch+1}/{self.epochs} [Val]", leave=False)
                    for users, target_item, neg_items in val_loop:
                        users, target_item, neg_items = users.to(self.device), target_item.to(self.device), neg_items.to(self.device)
                        
                        batch_size, num_negs = neg_items.shape
                        
                        pos_scores = self.model.predict(users, target_item)
                        neg_scores = self.model.predict(users, neg_items)
                        
                        loss = -torch.mean(torch.log(torch.sigmoid(pos_scores.unsqueeze(1) - neg_scores) + 1e-9))
                        val_total_loss += loss.item()
                        
                        all_scores = torch.cat([pos_scores.unsqueeze(1), neg_scores], dim=1)
                        _, top_indices = torch.topk(all_scores, k=10, dim=1)
                        
                        # 0번 인덱스(정답)가 topk에 포함되었는지 확인
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
            self.early_stopping.path # 모델 저장 경로를 기반으로 plot 경로 유추
        )

        # 최고 성능 모델 로드 후 반환
        self.model.load_state_dict(torch.load(self.early_stopping.path))
        return self.model