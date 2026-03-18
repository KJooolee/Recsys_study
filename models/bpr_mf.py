import torch
import torch.nn as nn

class BPRMF(nn.Module):
    def __init__(self, num_users, num_items, embed_dim=64):
        super(BPRMF, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embed_dim)
        self.item_embedding = nn.Embedding(num_items, embed_dim)
        
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

    def forward(self, users, pos_items, neg_items):
        """학습용: 정답과 오답 점수 동시 반환"""
        user_emb = self.user_embedding(users)
        pos_item_emb = self.item_embedding(pos_items)
        neg_item_emb = self.item_embedding(neg_items)
        
        pos_scores = (user_emb * pos_item_emb).sum(dim=1)
        neg_scores = (user_emb * neg_item_emb).sum(dim=1)
        
        return pos_scores, neg_scores

    def predict(self, users, items):
        """추론용: 특정 유저-아이템 점수 반환"""
        user_emb = self.user_embedding(users)
        item_emb = self.item_embedding(items)
        
        return (user_emb * item_emb).sum(dim=1)
    
    def get_all_item_embeddings(self):
        return self.item_embedding.weight.detach().cpu().numpy()