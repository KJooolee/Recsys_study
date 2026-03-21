import torch
import torch.nn as nn
from typing import List

class MLP(nn.Module):
    """
    TorchRec 공식 코드가 사용하는 Multi-Layer Perceptron (MLP) 모듈의 구조를 차용한 
    기본 파이토치 MLP 클래스입니다.
    """
    def __init__(self, in_size: int, layer_sizes: List[int]):
        super().__init__()
        layers = []
        for size in layer_sizes:
            layers.append(nn.Linear(in_size, size))
            layers.append(nn.ReLU())
            in_size = size
        self.model = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

class TwoTowerCL(nn.Module):
    """
    [Windows 구동 최적화용 Two-Tower 아키텍처]
    TorchRec의 공식 구조(Query, Candidate 타워 분리 및 독립형 MLP)를 그대로 차용하되,
    Windows(ctypes/DLL 오류) 및 BPRTrainer 환경(Sparse 연산)과 충돌하는
    무거운 EmbeddingBagCollection 의존성을 걷어낸 순수 파이토치 버전입니다.
    """
    def __init__(self, num_users: int, num_items: int, embed_dim: int = 64, layer_sizes: List[int] = [128, 64]):
        super().__init__()
        
        # 1. 임베딩 풀 (Torch.compile 최고 속도를 위해 dense embedding 사용)
        self.user_embedding = nn.Embedding(num_users, embed_dim)
        self.candidate_embedding = nn.Embedding(num_items, embed_dim)
        
        # 공식 권장 초기화
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.candidate_embedding.weight, std=0.01)
        
        # 2. 독립적인 두 개의 타워(Tower) 프로젝션
        self.query_proj = MLP(in_size=embed_dim, layer_sizes=layer_sizes)
        self.candidate_proj = MLP(in_size=embed_dim, layer_sizes=layer_sizes)

    def forward(self, users, pos_items, neg_items):
        """
        BPRTrainer의 학습 파이프라인(uses BPR Loss)에서 구동되기 위해 
        query(유저), pos(긍정 아이템), neg(부정 아이템)를 받아 점수를 계산합니다.
        """
        # A. Query(유저) 방향의 변환
        u_emb = self.query_proj(self.user_embedding(users))
        
        # B. Candidate(아이템) 방향의 변환
        pos_emb = self.candidate_proj(self.candidate_embedding(pos_items))
        neg_emb = self.candidate_proj(self.candidate_embedding(neg_items))
        
        # C. 추천 점수 스코어링 (기본적인 코사인 유사도/내적)
        pos_scores = (u_emb * pos_emb).sum(dim=1)
        neg_scores = (u_emb * neg_emb).sum(dim=1)
        
        # D. 오버피팅 억제용 Regularization (BPR-MF와 동일 속성)
        reg_loss = (1/2) * (u_emb.norm(2).pow(2) + pos_emb.norm(2).pow(2) + neg_emb.norm(2).pow(2)) / float(users.size(0))
        
        return pos_scores, neg_scores, reg_loss

    def predict(self, users, items):
        """
        Val/Test 페이즈에서 사용되는 1:N 랭킹 평가 지원 메서드입니다.
        """
        u_emb = self.query_proj(self.user_embedding(users))
        i_emb = self.candidate_proj(self.candidate_embedding(items))
        
        # 아이템 차원이 [B, 100] 등 음성 샘플 풀을 포함하고 있을 때
        if items.dim() == 2:
            return (u_emb.unsqueeze(1) * i_emb).sum(dim=-1)
        
        # 1:1 비교
        return (u_emb * i_emb).sum(dim=-1)

    def get_all_item_embeddings(self):
        """
        Test Evaluation 페이즈에서 Diversity(ILD) 등 아이템 간 거리(다양성) 지표 연산을 위해 호출됩니다.
        """
        device = self.candidate_embedding.weight.device
        num_items = self.candidate_embedding.num_embeddings
        items = torch.arange(num_items, device=device)
        return self.candidate_proj(self.candidate_embedding(items)).detach().cpu().numpy()
