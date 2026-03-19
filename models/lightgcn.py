import torch
import torch.nn as nn
import numpy as np
import scipy.sparse as sp

class LightGCN(nn.Module):
    def __init__(self, num_users, num_items, train_df, embed_dim=64, n_layers=3):
        """
        LightGCN: 딥러닝의 무거운 비선형 활성화 함수를 걷어내고, 
        오직 선형적인 이웃 노드 전파(Message Passing)만으로 추천 성능을 극대화한 그래프 모델.
        """
        super(LightGCN, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embed_dim = embed_dim
        self.n_layers = n_layers
        
        # 1. 초기 0번째 레이어 임베딩 ($E^{(0)}$)
        self.embedding_user = nn.Embedding(num_users, embed_dim)
        self.embedding_item = nn.Embedding(num_items, embed_dim)
        
        # 초기화 방법 (LightGCN 논문 권장 방식: Normal Distribution)
        nn.init.normal_(self.embedding_user.weight, std=0.1)
        nn.init.normal_(self.embedding_item.weight, std=0.1)
        
        # 2. 그래프 정규화 인접 행렬 생성 ($D^{-1/2} A D^{-1/2}$)
        self.norm_adj = self._build_graph(train_df)
        
    def _build_graph(self, train_df):
        print("\n🕸️ [LightGCN] 유저-아이템 그래프 인접 행렬(Adjacency Matrix) 구축 중...")
        
        users = train_df['user_id'].values
        items = train_df['item_id'].values
        interactions = np.ones(len(users))
        
        # 1. R 행렬 생성 (User-Item 상호작용)
        R = sp.coo_matrix((interactions, (users, items)), shape=(self.num_users, self.num_items))
        
        # 2. Bipartite Graph 인접 행렬 A 만들기
        # A = [0   R]
        #     [R^T 0]
        adj_mat = sp.bmat([[None, R], [R.T, None]])
        
        # 3. 정규화 (Degree 행렬을 이용해 연결선이 너무 많은 노드의 영향력 축소)
        rowsum = np.array(adj_mat.sum(axis=1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0. # 0으로 나누는 에러 방지
        d_mat = sp.diags(d_inv)
        
        norm_adj = d_mat.dot(adj_mat).dot(d_mat).tocoo()
        
        # 4. PyTorch 희소 행렬(Sparse Tensor)로 변환 (GPU 연산 최적화)
        indices = torch.LongTensor(np.vstack((norm_adj.row, norm_adj.col)))
        values = torch.FloatTensor(norm_adj.data)
        shape = torch.Size(norm_adj.shape)
        
        # 반드시 coalesce()를 호출해야 매 미니배치 sparse.mm 연산 시 내부 정렬(Sorting)로 인한 엄청난 속도 저하를 막을 수 있습니다.
        coo = torch.sparse_coo_tensor(indices, values, shape).coalesce()
        
        # 공식 레포(과거 PyTorch 버전)와 달리 최신 PyTorch의 CSR 포맷을 쓰면 메모리 접근 속도가 2~3배 빨라집니다.
        try:
            return coo.to_sparse_csr()
        except:
            return coo

    def get_embedding(self):
        """
        그래프 컨볼루션 레이어를 통과시키며 각 층의 임베딩을 평균내는 핵심 로직
        """
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        
        # $E^{(0)}$: 0번째 초기 임베딩 (유저와 아이템을 세로로 이어붙임)
        all_emb = torch.cat([users_emb, items_emb])
        embs = [all_emb]
        
        # K번의 Message Passing (이웃 정보 전파)
        # norm_adj가 GPU에 없다면 현재 모델이 위치한 디바이스로 보냄
        if self.norm_adj.device != all_emb.device:
            self.norm_adj = self.norm_adj.to(all_emb.device)
            
        for layer in range(self.n_layers):
            # $E^{(k+1)} = \tilde{A} E^{(k)}$ (행렬 곱셈 한 번으로 모든 이웃 정보 흡수)
            all_emb = torch.sparse.mm(self.norm_adj, all_emb)
            embs.append(all_emb)
            
        # 모든 레이어의 결과를 쌓아서 평균 (과적합 방지 및 다양한 수용범위 확보)
        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)
        
        # 다시 유저와 아이템 임베딩으로 쪼개서 반환
        users, items = torch.split(light_out, [self.num_users, self.num_items])
        return users, items

    def forward(self, users, pos_items, neg_items):
        """
        BPRTrainer와 완벽하게 호환되는 학습용 Forward Pass
        """
        all_users, all_items = self.get_embedding()
        
        user_emb = all_users[users]
        pos_item_emb = all_items[pos_items]
        neg_item_emb = all_items[neg_items]
        
        pos_scores = (user_emb * pos_item_emb).sum(dim=1)
        neg_scores = (user_emb * neg_item_emb).sum(dim=1)
        
        # LightGCN 공식 레포지토리의 정규화 기법 적용:
        # 모든 전파 레이어가 아니라 0번째 층(초기 임베딩)의 파라미터만 L2 Penalty(정규화) 측정
        user_emb0 = self.embedding_user(users)
        pos_emb0 = self.embedding_item(pos_items)
        neg_emb0 = self.embedding_item(neg_items)
        
        reg_loss = (1/2)*(user_emb0.norm(2).pow(2) + pos_emb0.norm(2).pow(2) + neg_emb0.norm(2).pow(2)) / float(len(users))
        
        return pos_scores, neg_scores, reg_loss

    def train(self, mode=True):
        """학습 모드 전환 시 평가용 캐시 초기화"""
        super().train(mode)
        if mode:
            self._eval_users_emb = None
            self._eval_items_emb = None
        return self

    def predict(self, users, items):
        """
        추론(평가) 시에 사용할 스코어링 함수 (평가 시 그래프 임베딩 캐싱 적용)
        """
        if self.training:
            all_users, all_items = self.get_embedding()
        else:
            if not hasattr(self, '_eval_users_emb') or self._eval_users_emb is None:
                self._eval_users_emb, self._eval_items_emb = self.get_embedding()
            all_users, all_items = self._eval_users_emb, self._eval_items_emb
        
        user_emb = all_users[users]
        item_emb = all_items[items]
        
        if items.dim() == 2:
            return (user_emb.unsqueeze(1) * item_emb).sum(dim=-1)
        return (user_emb * item_emb).sum(dim=-1)
    
    def get_all_item_embeddings(self):
        """Diversity(ILD) 계산을 위해 최종 전파된 아이템 임베딩 반환"""
        _, all_items = self.get_embedding()
        return all_items.detach().cpu().numpy()