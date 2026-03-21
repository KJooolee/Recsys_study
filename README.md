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
| **Deep Learning** | Two-Tower | 듀얼 인코더 및 독립 타워 구조를 차용한 대규모 피처 매칭 추천 | ✅ |

---

## 📈 Benchmark Results
`Clothing, Shoes & Jewelry` 데이터셋을 활용한 오프라인 평가(Leave-One-Out) 결과입니다.

| Model | Hit@10 | NDCG@10 | Coverage | Diversity (ILD) | Novelty | Serendipity |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **BPR-MF** | 0.5263 | 0.3489 | 0.9197 | 0.7478 | 17.2585 | 8.3688 |
| **LightGCN** | 0.5686 | 0.3845 | 0.9476 | 0.5624 | 17.8708 | 9.2688 |
| **SASRec** | 0.4938 | 0.3007 | 0.5346 | 0.9938 | 18.4878 | 8.9524 |
| **Two-Tower** | 0.4900 | 0.2998 | 0.2382 | 0.0550 | 16.5958 | 7.6028 |

> **💡 모델별 특성 요약:**
> - **LightGCN**은 희소성이 극단적인(99.99%) 환경 속에서도 그래프(Graph) 전파력을 발휘해 가장 높은 `Hit@10 (0.5686)` 및 높은 탐색 범위(`Coverage 0.9476`)를 달성하며 우수한 밸런스를 보였습니다.
> - **SASRec**은 희소 데이터 특성 상 행동 궤적이 단절되어 스코어는 다소 낮지만, 이색적인 아이템을 가장 잘 매칭하는 `Diversity (0.9938)` 및 `Novelty (18.48)` 지표에서 압도적인 잠재력을 보였습니다.
> - **Two-Tower**는 정확도(`Hit 0.49`) 측면에선 준수했으나, 인기도 편향(Popularity Bias)에 가장 심하게 노출되며 `Coverage (0.24)` 및 `Diversity (0.05)`가 붕괴되는 현상을 보였습니다. 이는 대용량 DNN 모델 특성상 별도의 Hard Negative Sampling 기법 없이는 소수의 인기 상품 공간에 표류하게 됨을 실증적으로 보여줍니다.

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
