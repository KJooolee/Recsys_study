import numpy as np
import torch
import os

class EarlyStopping:
    """
    Validation metric(예: NDCG, Hit Ratio)이 개선되지 않을 때 학습을 조기 종료합니다.
    """
    def __init__(self, patience=5, verbose=False, delta=0.0001, path='best_model.pt', mode='max'):
        """
        Args:
            patience (int): 성능이 개선되지 않아도 기다려주는 에포크 수
            verbose (bool): 개선 사항 출력 여부
            delta (float): 개선되었다고 판단할 최소한의 변화량
            path (str): 최고 성능 모델 가중치를 저장할 경로
            mode (str): 'max'면 지표가 커질수록 좋음(NDCG 등), 'min'이면 작을수록 좋음(Loss)
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        self.path = path
        self.mode = mode
        
        # 모델 저장 경로의 폴더가 없다면 생성
        os.makedirs(os.path.dirname(self.path), exist_ok=True)

    def __call__(self, current_score, model):
        # mode에 따라 점수 부호 조정 (최대화 문제 vs 최소화 문제)
        score = current_score if self.mode == 'max' else -current_score

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(current_score, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience} (Best: {self.best_score if self.mode == "max" else -self.best_score:.4f})')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(current_score, model)
            self.counter = 0

    def save_checkpoint(self, score, model):
        """성능이 개선되었을 때 모델을 저장합니다."""
        if self.verbose:
            print(f'Validation score improved. Saving model to {self.path} ...')
        torch.save(model.state_dict(), self.path)