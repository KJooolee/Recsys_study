import torch
from torch.utils.data import Dataset
import random
from collections import defaultdict
import numpy as np

class BPRTrainDataset(Dataset):
    """
    [학습용] BPR-MF 및 LightGCN을 위한 쌍체(Pairwise) 데이터셋
    반환 형태: (user, positive_item, negative_item)
    """
    def __init__(self, df_train, num_items):
        super(BPRTrainDataset, self).__init__()
        self.users = df_train['user_id'].values
        self.pos_items = df_train['item_id'].values
        self.num_items = num_items
        
        # 유저별로 이미 상호작용한 전체 아이템 셋 (빠른 네거티브 샘플링을 위해)
        self.user_pos_dict = self._build_user_pos_dict(df_train)

    def _build_user_pos_dict(self, df):
        user_pos = defaultdict(set)
        for user, item in zip(df['user_id'], df['item_id']):
            user_pos[user].add(item)
        return user_pos

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        user = self.users[idx]
        pos_item = self.pos_items[idx]
        
        while True:
            neg_item = random.randint(0, self.num_items - 1)
            if neg_item not in self.user_pos_dict[user]:
                break
                
        return torch.tensor(user, dtype=torch.long), \
               torch.tensor(pos_item, dtype=torch.long), \
               torch.tensor(neg_item, dtype=torch.long)

    def fast_sample(self, batch_size):
        """
        PyTorch DataLoader의 collate_fn 병목(개별 텐서를 수천 개씩 묶는 연산)을 피해
        LightGCN 공식 구현체처럼 Numpy 배열 단에서 통째로 썰어 매 에폭마다 텐서 배치를 고속 생성합니다.
        """
        n_samples = len(self.users)
        neg_items = np.empty(n_samples, dtype=np.int64)
        
        # 속도 최적화를 위한 numpy loop
        for i in range(n_samples):
            u = self.users[i]
            while True:
                neg = random.randint(0, self.num_items - 1)
                if neg not in self.user_pos_dict[u]:
                    break
            neg_items[i] = neg
            
        indices = np.random.permutation(n_samples)
        shuffled_users = self.users[indices]
        shuffled_pos = self.pos_items[indices]
        shuffled_neg = neg_items[indices]
        
        batches = []
        for start in range(0, n_samples, batch_size):
            end = start + batch_size
            if end > n_samples: break # drop_last=True 구조
            batches.append((
                torch.LongTensor(shuffled_users[start:end]),
                torch.LongTensor(shuffled_pos[start:end]),
                torch.LongTensor(shuffled_neg[start:end])
            ))
        return batches


class EvalDataset(Dataset):
    """
    [평가용] Leave-One-Out 검증/테스트용 데이터셋
    반환 형태: (user, target_item, negative_items_list)
    """
    def __init__(self, df_eval, df_train, num_items, num_negatives=99):
        """
        Args:
            df_eval: Val 또는 Test 데이터프레임
            df_train: Train 데이터프레임 (이미 본 아이템을 정답 후보에서 제외하기 위함)
            num_negatives: 타겟 아이템과 섞어놓을 네거티브 아이템의 수 (보통 99개 사용)
        """
        super(EvalDataset, self).__init__()
        self.users = df_eval['user_id'].tolist()
        self.target_items = df_eval['item_id'].tolist()
        self.num_items = num_items
        self.num_negatives = num_negatives
        
        # Train에서 이미 본 아이템은 추천 목록에서 배제해야 하므로 기록해둠
        self.train_user_pos_dict = BPRTrainDataset(df_train, num_items).user_pos_dict

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        user = self.users[idx]
        target_item = self.target_items[idx]
        
        # 평가를 위한 네거티브 아이템 N개 추출 (타겟 아이템 + Train에서 본 아이템 제외)
        neg_items = []
        while len(neg_items) < self.num_negatives:
            neg_item = random.randint(0, self.num_items - 1)
            if (neg_item not in self.train_user_pos_dict[user]) and (neg_item != target_item):
                neg_items.append(neg_item)
                
        return torch.tensor(user, dtype=torch.long), \
               torch.tensor(target_item, dtype=torch.long), \
               torch.tensor(neg_items, dtype=torch.long)