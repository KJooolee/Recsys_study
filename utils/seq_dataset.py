import torch
from torch.utils.data import Dataset
import random
from collections import defaultdict
import numpy as np

def get_user_seqs(df):
    """유저별로 시간순으로 정렬된 아이템 시퀀스 딕셔너리를 생성합니다."""
    user_seqs = defaultdict(list)
    for user, item in zip(df['user_id'], df['item_id']):
        # 패딩(0)과 겹치지 않게 하기 위해 실제 아이템 ID에 1을 더해줍니다.
        user_seqs[user].append(item + 1) 
    return user_seqs

class SeqTrainDataset(Dataset):
    """SASRec 학습을 위한 시퀀스 데이터셋 (Max Length만큼 잘라내고 빈칸은 0으로 채움)"""
    def __init__(self, df_train, num_items, max_len=50):
        self.user_seqs = get_user_seqs(df_train)
        self.users = list(self.user_seqs.keys())
        self.num_items = num_items
        self.max_len = max_len

    def __len__(self):
        return len(self.users)

    def __getitem__(self, index):
        user = self.users[index]
        seq = self.user_seqs[user]

        # max_len 크기의 빈 배열(0으로 채워짐) 준비
        inputs = np.zeros([self.max_len], dtype=np.int64)
        positives = np.zeros([self.max_len], dtype=np.int64)
        negatives = np.zeros([self.max_len], dtype=np.int64)

        nxt = seq[-1]
        idx = self.max_len - 1

        # 시퀀스를 뒤에서부터 역순으로 탐색하며 입력(Input)과 정답(Positive)을 만듭니다.
        for i in reversed(seq[:-1]):
            inputs[idx] = i
            positives[idx] = nxt
            
            # 오답(Negative) 랜덤 샘플링
            while True:
                neg = random.randint(1, self.num_items)
                if neg not in seq:
                    break
            negatives[idx] = neg
            
            nxt = i
            idx -= 1
            if idx == -1: break # max_len을 넘어가면 중단

        return torch.tensor(user, dtype=torch.long), \
               torch.tensor(inputs, dtype=torch.long), \
               torch.tensor(positives, dtype=torch.long), \
               torch.tensor(negatives, dtype=torch.long)

class SeqEvalDataset(Dataset):
    """SASRec 평가(Val/Test)를 위한 시퀀스 데이터셋"""
    def __init__(self, df_eval, df_train, num_items, max_len=50, num_negatives=99):
        self.user_seqs_train = get_user_seqs(df_train)
        self.users = df_eval['user_id'].tolist()
        # 타겟 아이템도 +1 처리
        self.target_items = (df_eval['item_id'] + 1).tolist() 
        self.num_items = num_items
        self.max_len = max_len
        self.num_negatives = num_negatives

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        user = self.users[idx]
        target_item = self.target_items[idx]
        seq = self.user_seqs_train[user]

        # 입력 시퀀스 패딩 처리 (가장 최근 max_len개만 사용)
        inputs = np.zeros([self.max_len], dtype=np.int64)
        seq_to_use = seq[-self.max_len:]
        inputs[-len(seq_to_use):] = seq_to_use

        # 평가용 네거티브 샘플링 (1~num_items 안에서)
        neg_items = []
        while len(neg_items) < self.num_negatives:
            neg_item = random.randint(1, self.num_items)
            if (neg_item not in seq) and (neg_item != target_item):
                neg_items.append(neg_item)

        return torch.tensor(user, dtype=torch.long), \
               torch.tensor(inputs, dtype=torch.long), \
               torch.tensor(target_item, dtype=torch.long), \
               torch.tensor(neg_items, dtype=torch.long)