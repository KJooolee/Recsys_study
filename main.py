# main.py
import os
import torch
import pandas as pd
from torch.utils.data import DataLoader
import torch.optim as optim

from models.bpr_mf import BPRMF
from trainers.bpr_trainer import BPRTrainer
from utils.dataset import BPRTrainDataset, EvalDataset

def load_and_remap_data(source):
    dataset_name_map = {
        'Clothing': 'Clothing_Shoes_and_Jewelry',
        'Electronics': 'Electronics',
        'Video_Games': 'Video_Games'
    }
    dataset_dir_name = dataset_name_map.get(source, source)
    data_dir = os.path.join('data', 'amazon', dataset_dir_name, 'temporal_split')
    
    print(f"[{source}] 데이터를 불러오는 중... ({data_dir})")
    train_df = pd.read_csv(os.path.join(data_dir, 'train.csv'))
    val_df = pd.read_csv(os.path.join(data_dir, 'val.csv'))
    test_df = pd.read_csv(os.path.join(data_dir, 'test.csv'))
    
    print("정수 ID로 변환 중...")
    user_ids = pd.concat([train_df['user_id'], val_df['user_id'], test_df['user_id']]).unique()
    item_ids = pd.concat([train_df['item_id'], val_df['item_id'], test_df['item_id']]).unique()
    
    user2id = {u: i for i, u in enumerate(user_ids)}
    item2id = {item: i for i, item in enumerate(item_ids)}
    
    train_df['user_id'] = train_df['user_id'].map(user2id)
    train_df['item_id'] = train_df['item_id'].map(item2id)
    
    val_df['user_id'] = val_df['user_id'].map(user2id)
    val_df['item_id'] = val_df['item_id'].map(item2id)
    
    test_df['user_id'] = test_df['user_id'].map(user2id)
    test_df['item_id'] = test_df['item_id'].map(item2id)
    
    num_users = len(user2id)
    num_items = len(item2id)
    
    return train_df, val_df, test_df, num_users, num_items, item2id

def run_pipeline(source, model_choice, mode_choice):
    """외부에서 변수를 주입받아 전체 파이프라인을 실행하는 메인 엔진"""
    
    train_df, val_df, test_df, num_users, num_items, item2id = load_and_remap_data(source)
    batch_size = 1024
    train_dataset = BPRTrainDataset(train_df, num_items)
    val_dataset = EvalDataset(val_df, train_df, num_items, num_negatives=99)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_name_map = {'1': 'BPR-MF', '2': 'LightGCN', '3': 'SASRec'}
    model_name = model_name_map.get(model_choice, 'BPR-MF')
    
    save_path = f'checkpoints/{source}_{model_name}_best.pt'
    os.makedirs('checkpoints', exist_ok=True)
    
    # 모델 선택
    if model_choice == '1':
        model = BPRMF(num_users, num_items, embed_dim=64)
    else:
        print(f"🚨 아직 구현되지 않은 모델입니다: {model_name}")
        return
        
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    # 실행 모드 분기
    if mode_choice == '1':
        print(f"\n=== [{model_name}] 학습을 시작합니다 ===")
        trainer = BPRTrainer(model, optimizer, device, epochs=50, save_path=save_path)
        trained_model = trainer.train_and_evaluate(train_loader, val_loader)
    elif mode_choice == '2':
        print(f"\n=== [{model_name}] 저장된 모델을 불러옵니다 ===")
        model.load_state_dict(torch.load(save_path, weights_only=True))

if __name__ == "__main__":
    main()