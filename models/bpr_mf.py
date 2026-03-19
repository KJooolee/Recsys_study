import torch.nn as nn

class BPRMF(nn.Module):
    def __init__(self, num_users, num_items, embed_dim=64):
        super(BPRMF, self).__init__()
        # sparse=True를 통해 수백만 개의 유저/아이템 그래디언트를 Dense(메모리 폭증)가 아닌 Sparse로 처리합니다
        self.user_emb = nn.Embedding(num_users, embed_dim, sparse=True)
        self.item_emb = nn.Embedding(num_items, embed_dim, sparse=True)
        
        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)
        
    def forward(self, users, pos_items, neg_items):
        u_emb = self.user_emb(users)
        pos_emb = self.item_emb(pos_items)
        neg_emb = self.item_emb(neg_items)
        
        pos_scores = (u_emb * pos_emb).sum(dim=1)
        neg_scores = (u_emb * neg_emb).sum(dim=1)
        
        # O(1) 정규화를 위한 L2 Regularization 수동 계산
        reg_loss = (1/2)*(u_emb.norm(2).pow(2) + pos_emb.norm(2).pow(2) + neg_emb.norm(2).pow(2)) / float(len(users))
        
        return pos_scores, neg_scores, reg_loss
        
        pos_scores = (u_emb * pos_emb).sum(dim=1)
        neg_scores = (u_emb * neg_emb).sum(dim=1)
        
        return pos_scores, neg_scores
        
    def predict(self, users, items):
        u_emb = self.user_emb(users)
        i_emb = self.item_emb(items)
        if items.dim() == 2:
            return (u_emb.unsqueeze(1) * i_emb).sum(dim=-1)
        return (u_emb * i_emb).sum(dim=-1)
        
    def get_all_item_embeddings(self):
        return self.item_emb.weight.detach().cpu().numpy()