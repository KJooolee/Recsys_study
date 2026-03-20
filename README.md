# Recsys_study

단순한 모델의 성능(Accuracy) 경쟁을 넘어, 실제 유저의 행동 데이터가 가진 한계를 분석하고 다양성(Diversity)과 우연한 발견(Serendipity)을 촉진할 수 있는 최적의 머신러닝 구조를 탐구하는 저장소입니다.

## Project Overview
- **Objective:** 이커머스/리테일 도메인의 현실적인 데이터 제약(Sparsity, Popularity Bias)을 분석하고, 이를 해결하기 위한 다양한 패러다임의 추천 모델을 직접 구현하여 비교 분석합니다.
- **Key Focus:** 
  - 특정 기술에 매몰되지 않고 데이터 특성(Graph vs Sequence)에 맞는 모델 매칭
  - 정확도 지표(NDCG, Hit Ratio) 외에 콜드 스타트 커버리지 및 참신성(Novelty) 검증

---

## 📊 EDA: Amazon Clothing, Shoes & Jewelry Dataset
현재 벤치마크에 사용 중인 핵심 데이터셋입니다. 
- **Users (유저 수):** 1,219,678 명
- **Items (아이템 수):** 376,139 개
- **Interactions (상호작용 수):** 8,846,108 건
- **Sparsity (희소성):** 99.9981%

> **💡 Insight:** 유저와 아이템 수가 워낙 방대하고 희소성이 99.99%를 초과하는 극단적 `Cold-start` 및 `Long-tail` 특성을 가집니다. 추후 해당 문제를 비교 대조하기 위해, 크기가 작고 분포가 다른(Dense한) 신규 데이터셋들을 추가 연동하여 실험할 예정입니다.

---

## 🛠 Models Implemented
추천 시스템의 뼈대가 되는 핵심 방향성 모델들을 순차적으로 구현하고 최적화합니다.

| 패러다임 | 알고리즘 | 목적 및 가설 | Status |
| :--- | :--- | :--- | :---: |
| **Latent Factor** | BPR-MF | 암시적 피드백(Implicit Feedback) 환경의 순위 학습(Pairwise) 베이스라인 구축 | ✅ |
| **Graph** | LightGCN | 고차원 이웃 정보(High-order Connectivity) 전파를 통한 희소성/콜드스타트 극복 | ✅ |
| **Sequential** | SASRec | Self-Attention(Pre-LayerNorm 등 공식 구현체 최적화 적용)을 활용한 행동의 시간적 맥락 예측 | ✅ |
| **Autoencoder** | Multi-VAE | 유저 취향의 비선형적 패턴 학습 및 다항 분포 기반 생성형 추천 | ⏳ |

*(Multi-VAE은 순차적으로 연구 및 적용될 예정입니다.)*

---

## 📈 Benchmark Results
`Clothing, Shoes & Jewelry` 데이터셋을 활용한 오프라인 평가(Leave-One-Out) 결과입니다.

| Model | Hit@10 | NDCG@10 | Coverage | Diversity (ILD) | Novelty | Serendipity |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **BPR-MF** | 0.5263 | 0.3489 | 0.9197 | 0.7478 | 17.2585 | 8.3688 |
| **LightGCN** | 0.5686 | 0.3845 | 0.9476 | 0.5624 | 17.8708 | 9.2688 |
| **SASRec** | 0.4938 | 0.3007 | 0.5346 | 0.9938 | 18.4878 | 8.9524 |

> **💡 모델별 특성 요약:**
> - **LightGCN**은 희소성이 99.99%에 달하는 현재 데이터셋에서 그래프(Graph) 전파력을 발휘해 가장 높은 `Hit@10 (0.5686)` 및 가장 완벽에 가까운 추천 범위(`Coverage 0.9476`)를 달성했습니다.
> - **SASRec**은 희소 데이터 특성 상 앞선 행동 맥락이 자주 단절되어 정확도는 가장 낮았으나, 유저에게 매우 이색적인 아이템을 섞어서 제시하는 `Diversity (0.9938)` 및 `Novelty (18.48)` 지표에서 매우 우수한 특성을 보였습니다.

---

## 📂 Repository Structure
```text
├── data/                  # 실험에 활용한 데이터 파일
├── notebooks/             # 데이터 EDA 분석 및 분할을 위한 노트북 파일
├── models/                # 각 추천 알고리즘별 PyTorch 커스텀 최적화 구현체 (SASRec, LightGCN 등)
├── trainers/              # O(1) Numpy 고속화 샘플러 및 Early Stopping 검증 파이프라인
├── results/               # 모델별 평가지표 및 학습/검증 메트릭 시각화 결과물
├── utils/                 # 평가지표(Metrics) 연산, 데이터 로더 분기 등
└── README.md
```
