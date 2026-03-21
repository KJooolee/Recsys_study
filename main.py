import os
import torch
import pandas as pd
from torch.utils.data import DataLoader
import torch.optim as optim

from models.bpr_mf import BPRMF
from models.lightgcn import LightGCN
from models.sasrec import SASRec
from models.two_tower_cl import TwoTowerCL

from trainers.bpr_trainer import BPRTrainer
from trainers.seq_trainer import SeqTrainer

from utils.dataset import BPRTrainDataset, EvalDataset
from utils.seq_dataset import SeqTrainDataset, SeqEvalDataset
from evaluate import evaluate_model

def load_and_remap_data(source):
    data_dir = f'data/amazon/{source}/temporal_split'
    print(f"\n데이터 로딩 및 정수 ID 변환 중... ({data_dir})")
    try:
        train_df = pd.read_csv(os.path.join(data_dir, 'train.csv'))
        val_df = pd.read_csv(os.path.join(data_dir, 'val.csv'))
        test_df = pd.read_csv(os.path.join(data_dir, 'test.csv'))
    except FileNotFoundError:
        print(f"🚨 에러: {data_dir} 경로에 데이터가 없습니다. 먼저 02_Data_Split.ipynb를 실행해주세요.")
        raise
    
    all_users = pd.concat([train_df['user_id'], val_df['user_id'], test_df['user_id']]).unique()
    all_items = pd.concat([train_df['item_id'], val_df['item_id'], test_df['item_id']]).unique()
    
    user2id = {u: i for i, u in enumerate(all_users)}
    item2id = {i: idx for idx, i in enumerate(all_items)}
    
    for df in [train_df, val_df, test_df]:
        df['user_id'] = df['user_id'].map(user2id)
        df['item_id'] = df['item_id'].map(item2id)
        
    num_users = len(user2id)
    num_items = len(item2id)
    print(f"✅ 변환 완료! (총 유저: {num_users:,}명 / 총 아이템: {num_items:,}개)")
    
    return train_df, val_df, test_df, num_users, num_items, item2id

def run_pipeline(source, model_choice, mode_choice):
    # 1. 데이터 준비
    train_df, val_df, test_df, num_users, num_items, item2id = load_and_remap_data(source)
    
    print("\nDataLoader를 세팅합니다...")
    if model_choice == '1' or model_choice == '4':
        batch_size = 8192     
    elif model_choice == '2':
        batch_size = 32768      
    elif model_choice == '3': 
        batch_size = 2048     
    
    if model_choice in ['1', '2', '4']: # BPR-MF, LightGCN, TwoTowerAdapter (단일 아이템 기반 모델들)
        train_dataset = BPRTrainDataset(train_df, num_items)
        val_dataset = EvalDataset(val_df, train_df, num_items, num_negatives=99)
        test_dataset = EvalDataset(test_df, train_df, num_items, num_negatives=99)
    elif model_choice == '3':      # SASRec (시퀀스 리스트)
        train_dataset = SeqTrainDataset(train_df, num_items, max_len=50)
        val_dataset = SeqEvalDataset(val_df, train_df, num_items, max_len=50, num_negatives=99)
        test_dataset = SeqEvalDataset(test_df, train_df, num_items, max_len=50, num_negatives=99)
    
    train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            drop_last=True,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True
        )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)
    
    # 3. 장치 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"✅ 사용 디바이스: {device}")
    
    model_name_map = {'1': 'BPR-MF', '2': 'LightGCN', '3': 'SASRec', '4': 'TwoTower'}
    model_name = model_name_map.get(model_choice, 'BPR-MF')
    
    save_path = f'checkpoints/{source}_{model_name}_best.pt'
    os.makedirs('checkpoints', exist_ok=True)
    
    if model_choice == '1':
        model = BPRMF(num_users, num_items, embed_dim=64)
        # BPR-MF는 Sparse 옵티마이저를 사용해 1에폭당 18분 걸리는 메모리 할당/업데이트 병목을 10초 내외로 단축합니다
        optimizer = optim.SparseAdam(model.parameters(), lr=0.001)
    elif model_choice == '2':
        model = LightGCN(num_users, num_items, train_df, embed_dim=64, n_layers=3)
        # LightGCN은 파이토치 옵티마이저의 범용 weight_decay를 켤 경우 수백만 개 임베딩 전체가 통째로 0으로 수렴하는 치명적 붕괴가 일어납니다.
        # 따라서 weight_decay=0.0으로 끄고, 모델 내부에서 직접 추출한 배치 수치(reg_loss)에만 Penalty를 더합니다.
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0)
    elif model_choice == '3':
        # SASRec 초기화
        model = SASRec(num_items, max_len=50, embed_dim=64, num_heads=2, num_blocks=2, dropout_rate=0.2)
        # 공식 pmixer/SASRec.pytorch와 마찬가지로 전역 weight_decay를 0으로 해제하여 Hit=1 매몰 버그를 막습니다.
        optimizer = optim.Adam(model.parameters(), lr=0.001)
    elif model_choice == '4':
        model = TwoTowerCL(num_users, num_items, embed_dim=64)
        optimizer = optim.Adam(model.parameters(), lr=0.001)



    
    if mode_choice == '1':
        print(f"\n=== [{model_name}] 학습을 시작합니다 ===")
        
        if model_choice in ['1', '2', '4']:
            trainer = BPRTrainer(model, optimizer, device, epochs=50, save_path=save_path)
        elif model_choice == '3':
            trainer = SeqTrainer(model, optimizer, device, epochs=50, save_path=save_path)
            
        trained_model = trainer.train_and_evaluate(train_loader, val_loader)
        
        print(f"\n✅ 학습 완료! 이어서 Test 셋으로 최종 평가를 진행합니다.")
        # 모델 구분 없이 무조건 평가 실행!
        evaluate_model(trained_model, test_loader, train_df, num_items, device, k=10, source=source, model_name=model_name)
            
    elif mode_choice == '2':
        print(f"\n=== [{model_name}] 저장된 모델을 불러옵니다 ===")
        if os.path.exists(save_path):
            model.load_state_dict(torch.load(save_path, weights_only=True))
            print("✅ 체크포인트 로드 완료! Test 셋 평가를 진행합니다.")
            
            # 모델 구분 없이 무조건 평가 실행!
            evaluate_model(model, test_loader, train_df, num_items, device, k=10, source=source, model_name=model_name)
        else:
            print(f"🚨 에러: {save_path} 파일이 존재하지 않습니다. 먼저 1번 모드(학습)를 진행해주세요.")    

if __name__ == "__main__":
    main()