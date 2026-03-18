import torch
import torch.nn as nn

class SASRec(nn.Module):
    def __init__(self, num_items, max_len=50, embed_dim=64, num_heads=2, num_blocks=2, dropout_rate=0.2):
        """
        SASRec: 유저의 과거 행동 '시퀀스(순서)'를 Transformer Encoder로 분석하여 
        다음 행동을 예측하는 강력한 시퀀셜 모델.
        """
        super(SASRec, self).__init__()
        self.num_items = num_items
        self.max_len = max_len
        
        # 🚨 중요: 시퀀스 모델은 길이가 모자랄 때 빈 공간을 채우는 '패딩(Padding)'이 필요합니다.
        # 보통 0번을 패딩 인덱스로 사용하므로, 실제 아이템 임베딩은 num_items + 1 개를 만듭니다.
        self.item_emb = nn.Embedding(self.num_items + 1, embed_dim, padding_idx=0)
        self.pos_emb = nn.Embedding(self.max_len, embed_dim) # 위치(시간) 정보 임베딩
        self.emb_dropout = nn.Dropout(dropout_rate)
        
        # Transformer Encoder 뼈대 생성
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout_rate,
            activation='relu',
            batch_first=True # [batch_size, seq_len, embed_dim] 형태 유지
        )
        # 여러 층(num_blocks)을 쌓아서 깊은 맥락을 파악
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_blocks)
        
    def forward(self, seqs):
        """
        입력된 아이템 시퀀스에 대해 Self-Attention을 수행하고 숨겨진 맥락(Hidden States)을 반환합니다.
        - seqs: [batch_size, max_len] 형태의 유저 과거 상호작용 아이템 ID 리스트
        """
        batch_size, seq_len = seqs.size()
        
        # 1. 포지셔널 임베딩 생성 (최신 데이터일수록 다른 위치값을 가지도록 함)
        positions = torch.arange(seq_len, dtype=torch.long, device=seqs.device)
        positions = positions.unsqueeze(0).expand_as(seqs) # [batch_size, max_len]
        
        # 2. 아이템 임베딩 + 포지셔널 임베딩 더하기
        seq_embs = self.item_emb(seqs) + self.pos_emb(positions)
        seq_embs = self.emb_dropout(seq_embs)
        
        # 3. 마스킹 (Masking) 설계
        # A. 패딩 마스크: 0으로 채워진 가짜 아이템(패딩)은 어텐션에서 제외 (True가 무시됨)
        padding_mask = (seqs == 0) # [batch_size, max_len]
        
        # B. Causal Mask: 과거를 보고 '미래'를 컨닝하지 못하게 대각선 위쪽을 가림
        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(seqs.device)
        
        # 4. Transformer 통과
        log_feats = self.transformer(seq_embs, mask=causal_mask, src_key_padding_mask=padding_mask)
        
        return log_feats # [batch_size, max_len, embed_dim]

    def predict(self, log_feats, item_indices):
        """
        [추론/평가용] 시퀀스의 가장 마지막 은닉 상태(Last Hidden State)를 사용하여
        후보 아이템들에 대한 추천 점수를 계산합니다.
        """
        # 가장 마지막 시점(가장 최신 맥락)의 벡터만 추출 -> [batch_size, embed_dim]
        final_feat = log_feats[:, -1, :] 
        
        # 타겟 아이템들의 임베딩 추출
        item_embs = self.item_emb(item_indices)
        
        # 내적을 통해 최종 점수 산출
        if item_embs.dim() == 2: # 1D 텐서 (모든 아이템 예측)
            scores = torch.matmul(final_feat, item_embs.transpose(0, 1))
        else: # 2D 텐서 (네거티브 샘플링된 아이템들 예측)
            scores = (final_feat.unsqueeze(1) * item_embs).sum(dim=-1)
            
        return scores
        
    def get_all_item_embeddings(self):
        """Diversity(ILD) 계산을 위한 헬퍼 함수 (패딩 0번 제외)"""
        return self.item_emb.weight[1:].detach().cpu().numpy()