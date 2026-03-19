import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
import random

class BPRDataset(Dataset):
    """
    BPR-MF 및 LightGCN 등 Graph/Matrix 기반 모델을 위한 Dataset
    Triplet 형식 반환: (user, positive_item, negative_item)
    """
    def __init__(self, df_train, num_items):
        super(BPRDataset, self).__init__()
        self.df_train = df_train
        self.num_items = num_items
        
        # 유저별로 긍정적 상호작용(구매/클릭)을 한 아이템 목록을 Set으로 저장
        # 이유: 네거티브 샘플링 시, 이미 구매한 아이템을 뽑지 않기 위함
        self.user_pos_items = self._get_user_pos_items()
        
        # DataLoader에서 index로 접근하기 위해 리스트로 변환
        self.users = self.df_train['user_id'].tolist()
        self.pos_items = self.df_train['item_id'].tolist()

    def _get_user_pos_items(self):
        user_pos_items = defaultdict(set)
        for _, row in self.df_train.iterrows():
            user_pos_items[row['user_id']].add(row['item_id'])
        return user_pos_items

    def __len__(self):
        return len(self.df_train)

    def __getitem__(self, idx):
        user = self.users[idx]
        pos_item = self.pos_items[idx]
        
        # 네거티브 샘플링: 유저가 상호작용하지 않은 아이템을 랜덤하게 1개 뽑음
        neg_item = self._sample_negative(user)
        
        # PyTorch 텐서로 변환하여 반환
        return torch.tensor(user, dtype=torch.long), \
               torch.tensor(pos_item, dtype=torch.long), \
               torch.tensor(neg_item, dtype=torch.long)

    def _sample_negative(self, user):
        while True:
            # 0부터 num_items-1 사이에서 랜덤하게 하나 선택
            neg_item = random.randint(0, self.num_items - 1)
            # 만약 뽑은 아이템이 유저가 이미 상호작용한 아이템이 아니라면 채택
            if neg_item not in self.user_pos_items[user]:
                return neg_item

def get_dataloader(train_path, num_items, batch_size=1024, num_workers=0):
    """
    학습용 DataLoader를 생성하여 반환하는 헬퍼 함수
    """
    df_train = pd.read_csv(train_path)
    
    dataset = BPRDataset(df_train, num_items)
    
    # shuffle=True는 필수: 데이터를 섞어주어야 모델이 순서에 편향되지 않음
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        drop_last=True # 배치 사이즈에 딱 안 맞고 남는 찌꺼기 데이터는 버림 (학습 안정화)
    )
    
    return dataloader, dataset.user_pos_items