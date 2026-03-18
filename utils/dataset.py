import torch
from torch.utils.data import Dataset
import random
from collections import defaultdict

class BPRTrainDataset(Dataset):
    """
    [학습용] BPR-MF 및 LightGCN을 위한 쌍체(Pairwise) 데이터셋
    반환 형태: (user, positive_item, negative_item)
    """
    def __init__(self, df_train, num_items):
        super(BPRTrainDataset, self).__init__()
        self.users = df_train['user_id'].tolist()
        self.pos_items = df_train['item_id'].tolist()
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
        
        # 유저가 한 번도 클릭하지 않은 아이템을 1개 랜덤 샘플링
        while True:
            neg_item = random.randint(0, self.num_items - 1)
            if neg_item not in self.user_pos_dict[user]:
                break
                
        return torch.tensor(user, dtype=torch.long), \
               torch.tensor(pos_item, dtype=torch.long), \
               torch.tensor(neg_item, dtype=torch.long)


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