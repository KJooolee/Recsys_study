# Recsys_study

단순한 모델의 성능(Accuracy) 경쟁을 넘어, 실제 유저의 행동 데이터가 가진 한계를 분석하고 **'다양성(Diversity)'**과 **'우연한 발견(Serendipity)'**을 촉진할 수 있는 최적의 머신러닝 구조를 탐구하는 저장소입니다.

## Project Overview
- **Objective:** 이커머스/리테일 도메인의 현실적인 데이터 제약(Sparsity, Popularity Bias)을 분석하고, 이를 해결하기 위한 다양한 패러다임의 추천 모델을 직접 구현하여 비교 분석합니다.
- **Key Focus:** - 특정 기술에 매몰되지 않고 데이터 특성(Graph vs Sequence)에 맞는 모델 매칭
  - 정확도 지표(NDCG, Hit Ratio) 외에 콜드 스타트 커버리지 및 참신성(Novelty) 검증
  - 오프라인 평가의 한계를 인지하고, 온라인 A/B 테스트(CTR, CVR)로 연결하기 위한 가설 수립

## Datasets (Amazon Review Data 2018)
다양한 유저 소비 패턴을 분석하기 위해 성격이 다른 카테고리를 비교합니다.
1. **Electronics:** 뚜렷한 목적성과 순차적 구매 패턴(Sequential) 분석
2. **Clothing, Shoes & Jewelry:** 강한 개인 취향 반영 및 극도의 희소성(Sparsity) 분석
3. **Video Games:** 특정 매니아층의 군집(Niche) 및 잠재 요인 분석

## 🛠️ Models Implemented
추천 시스템의 뼈대가 되는 5가지 핵심 방향성을 순차적으로 구현합니다.

| 패러다임 | 알고리즘 | 목적 및 가설 | Status |
| :--- | :--- | :--- | :---: |
| **Latent Factor** | BPR-MF | 암시적 피드백(Implicit Feedback) 환경의 랭킹 학습 베이스라인 구축 | ⬜️ |
| **Autoencoder** | Multi-VAE | 유저 취향의 비선형적 패턴 학습 및 생성형 추천 | ⬜️ |
| **Two-Tower** | DSSM | 대규모 아이템 환경에서의 실시간 후보 생성(Candidate Generation) 연산 효율 확보 | ⬜️ |
| **Graph** | LightGCN | 고차원 이웃 정보(High-order Connectivity)를 통한 롱테일/콜드스타트 아이템 발굴 | ⬜️ |
| **Sequential** | SASRec | Self-Attention을 활용한 유저 행동의 시간적 맥락 및 다음 행동 예측 | ⬜️ |

## Evaluation Metrics
현업의 비즈니스 요구사항을 반영하여 다각도로 모델을 평가합니다.
- **Accuracy:** NDCG@K, Hit Ratio@K
- **Business Impact:** - `Diversity (ILD)`: 추천 목록의 다양성 보장 (필터 버블 방지)
  - `Novelty & Serendipity`: 인기 편향(Popularity Bias) 완화 및 우연한 발견 유도
  - `Coverage`: 롱테일(Long-tail) 및 신규 아이템 추천 비율

## 📂 Repository Structure
```text
├── data/                  # 데이터 전처리 및 EDA 스크립트
├── models/                # 각 추천 알고리즘별 PyTorch/TensorFlow 구현체
├── notebooks/             # 가설 수립, EDA, 모델 결과 비교 등 분석 리포트 (Jupyter)
├── utils/                 # 평가지표(Metrics) 계산 및 데이터 로더 모듈
└── README.md
