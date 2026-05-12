# 신용점수 분류 딥러닝 컴페티션

> PyTorch MLP + XGBoost + LightGBM **3-모델 가중 앙상블**로 고객의 금융 행태 데이터로부터 신용등급(Good / Standard / Poor)을 분류하는 프로젝트입니다. 기본 베이스라인(71.52%) 대비 hold-out test 정확도 약 83%를 달성했습니다.
<br>

## 📑 목차

1. [프로젝트 개요](#-프로젝트-개요)
2. [개발 환경](#-개발-환경)
3. [데이터셋](#-데이터셋)
4. [전처리](#-전처리)
5. [EDA](#-eda)
6. [모델링 & 학습 전략](#-모델링--학습-전략)
7. [성능 결과](#-성능-결과)
8. [개선 포인트 요약](#-개선-포인트-요약)
9. [실행 방법](#-실행-방법)
10. [한계 및 향후 과제](#️-한계-및-향후-과제)

<br>

## 📌 프로젝트 개요

| 항목 | 내용 |
|---|---|
| **Task** | Multi-class Classification (Good / Standard / Poor) |
| **데이터** | [Kaggle - Credit Score Classification](https://www.kaggle.com/datasets/parisrohan/credit-score-classification) |
| **샘플 수** | 100,000 rows × 28 columns |
| **프레임워크** | PyTorch + XGBoost + LightGBM |
| **평가 지표** | Hold-out Test Accuracy |
| **목표 성능** | 75% 이상 |
| **달성 성능** | **약 87~89% (3-모델 가중 앙상블)** |

<br>

## 🔧 개발 환경

- Python 3.10+
- PyTorch ≥ 2.0
- scikit-learn ≥ 1.3
- xgboost, lightgbm
- pandas, numpy, matplotlib, seaborn
- Google Colab (GPU: T4)

```bash
pip install torch scikit-learn xgboost lightgbm pandas numpy matplotlib seaborn
```

<br>

## 📊 데이터셋

- 한 고객당 **8개월치 금융 정보**가 시계열로 들어있는 **패널 데이터** 구조
- 고유 고객 수: **12,500명 × 8개월 = 100,000행**
- 클래스 분포: Standard 53.2% / Poor 29.0% / Good 17.8% (불균형이지만 극단적이지 않음)

### 주요 컬럼 설명

| 컬럼 | 설명 |
|---|---|
| `Annual_Income` | 연 소득 |
| `Monthly_Inhand_Salary` | 월 실수령 급여 |
| `Outstanding_Debt` | 미지급 잔액 |
| `Num_of_Delayed_Payment` | 평균 연체 횟수 |
| `Delay_from_due_date` | 평균 연체 일수 |
| `Credit_Mix` | 신용 혼합 유형 (Good / Standard / Bad) |
| `Type_of_Loan` | 보유 대출 종류 (다중값, 콤마 구분) |
| `Credit_History_Age` | 신용 이력 연수 |
| `Credit_Score` | **타겟** (Good / Standard / Poor) |

<br>

## 🛠 전처리

### 1️⃣ 식별자 처리 + Customer_ID 활용

`ID`, `Name`, `SSN`은 예측에 무의미한 고유값이라 제거. 단, `Customer_ID`는 *제거 직전에* groupby에 활용해 **고객별 과거 8개월치 통계량**을 피처로 생성합니다.

```python
agg_cols = ['Annual_Income', 'Outstanding_Debt', 'Num_of_Delayed_Payment', ...]
for col in agg_cols:
    data[col + '_mean'] = data.groupby('Customer_ID')[col].transform('mean')
    data[col + '_std']  = data.groupby('Customer_ID')[col].transform('std').fillna(0)
```

→ "이번 달 부채"보다 "이 고객의 평균 부채와 변동성"이 신용도를 훨씬 잘 설명.

### 2️⃣ Type_of_Loan — Multi-hot Encoding

원본은 `"auto loan, student loan, ..."` 형식의 다중값 문자열(unique 6,261개). LabelEncoder로 묶으면 의미 없는 조합에 임의의 순서를 부여하게 됩니다.
→ 8개 주요 대출 유형 각각의 보유 여부를 0/1 컬럼으로 분리

### 3️⃣ 다중공선성 제거

`Annual_Income` ↔ `Monthly_Inhand_Salary` = **0.998** → 사실상 동일 정보. `Monthly_Inhand_Salary` 제거.

### 4️⃣ 비율 피처 생성

```python
data['Debt_Income_Ratio']   = data['Outstanding_Debt']             / (data['Annual_Income'] + 1)
data['EMI_Income_Ratio']    = data['Total_EMI_per_month'] * 12     / (data['Annual_Income'] + 1)
data['Invest_Income_Ratio'] = data['Amount_invested_monthly'] * 12 / (data['Annual_Income'] + 1)
```

### 5️⃣ 데이터 누수 차단

- `stratify=y`로 클래스 비율 유지하며 80:20 hold-out split
- `StandardScaler`는 **fold 내부에서 train fold에만 fit**, val/test에는 transform만

<br>

## 📈 EDA

> 모든 시각화는 코드 실행 시 `images/` 폴더에 자동 저장됩니다.

### 1) 클래스 분포

<img width="1317" height="528" alt="01_class_distribution" src="https://github.com/user-attachments/assets/97dd99dc-508d-4b63-9363-29950dc36e00" />

Standard 53.2% / Poor 29.0% / Good 17.8% — **불균형이지만 극단적이지 않음**. 실험에서 `class_weight`를 적용하면 소수 클래스(Good) recall은 오르지만 전체 accuracy가 약 1%p 떨어졌습니다. 평가 지표(accuracy) 기준에서는 가중치 미적용이 유리.

### 2) Mutual Information 상위 피처

<img width="1187" height="708" alt="02_mutual_information" src="https://github.com/user-attachments/assets/c656dcf1-5124-462f-8e43-d4d1a8386778" />

**소득·부채 절대값이 1군**(MI ≈ 0.6대): `Annual_Income`, `Monthly_Inhand_Salary`, `Amount_invested_monthly`, `Outstanding_Debt`, `Total_EMI_per_month`. `Annual_Income`과 `Monthly_Inhand_Salary`가 둘 다 상위 → 다중공선성 의심 → 아래 상관 히트맵에서 직접 확인.

흥미로운 점은 `Num_of_Delayed_Payment`보다 `Annual_Income` 같은 절대값 변수가 더 높은 MI를 가진다는 것 → **비율 피처(부채/소득)를 추가하면 시너지가 날 신호**.

### 3) 상관 히트맵

<img width="1045" height="946" alt="03_correlation_heatmap" src="https://github.com/user-attachments/assets/227edea7-55b1-4808-b78d-07a31f174193" />

확인된 사항:
- **`Annual_Income` ↔ `Monthly_Inhand_Salary` = 1.00** → 사실상 동일. `Monthly_Inhand_Salary` 제거의 직접적 근거.
- `Outstanding_Debt`, `Num_of_Delayed_Payment`, `Delay_from_due_date`, `Interest_Rate`가 서로 **0.5~0.6대로 강하게 묶임** → "재정 부담군" 클러스터 형성.

### 4) 주요 피처 vs Credit_Score

<img width="1549" height="947" alt="04_features_vs_target" src="https://github.com/user-attachments/assets/91693c19-2c0c-4a8f-854a-99aa6a947122" />

- **`Num_of_Delayed_Payment`, `Delay_from_due_date`**: Good → Poor로 갈수록 명확히 증가. 강한 분리력.
- **`Outstanding_Debt`**: Poor의 중앙값이 Good의 2배 이상.
- **`Annual_Income`**: 분포 겹침이 큼. 단독으로는 약한 분리력 → **비율 피처가 필요한 이유**.

### 5) 대출 유형별 신용 등급 분포

<img width="1309" height="647" alt="05_loan_type_distribution" src="https://github.com/user-attachments/assets/65da79a9-fb2a-42a6-bc09-d052f39b64db" />

**사전 추측과 다른 발견**: 8개 대출 유형 모두에서 신용 등급 분포가 거의 동일했습니다. 즉 **어떤 종류의 대출을 가졌는지보다 *몇 개를 어떻게 관리하는지*가 중요**합니다. 그럼에도 Multi-hot 컬럼을 유지한 이유는, 단독으로는 약하지만 다른 피처와의 상호작용 학습에 기여할 수 있기 때문입니다.

→ EDA는 가설을 확인하는 작업일 뿐 아니라 **반증**하는 도구이기도 합니다.

### 6) 패널 구조 확인

<img width="1429" height="528" alt="06_customer_panel" src="https://github.com/user-attachments/assets/5fd2304b-71ce-4337-8caa-cabd4ed078db" />

- 12,500명의 고객이 **모두 정확히 8개월씩** 등장 → 강한 패널 구조
- 같은 고객의 Credit_Score는 1개로 일관된 경우 **5,208명**, 2개 클래스가 섞인 경우 **7,262명**, 3개 다 등장한 경우 30명
- **같은 고객이라도 시점에 따라 신용 등급이 변동** → 단일 시점 값보다 고객별 평균/표준편차 같은 집계 통계가 노이즈를 줄여 더 안정적

<br>

## 🧠 모델링 & 학습 전략

### 모델 후보 비교

| 후보 | 채택 여부 | 사유 |
|---|---|---|
| **TabTransformer** | ❌ | LabelEncoding 후 거의 수치형 → 어텐션 이득 미미 |
| **TabNet** | ❌ | 구현 복잡도 대비 MLP와 큰 차이 없음 |
| **기본 MLP** (64→32→3) | ❌ | 표현 용량 부족 (71%대 정체) |
| **MLP + Residual** | ✅ | [512,256,128,64] + Skip Connection |
| **XGBoost** | ✅ | 정형 데이터 강자. 단독 83%+ |
| **LightGBM** | ✅ | XGB 보완용. 단독 82%+ |

### 핵심 전략: 3-모델 가중 앙상블 + 5-Fold CV

```
Fold 1 ──┬── MLP    ─────────┐
         ├── XGBoost  ───────┤
         └── LightGBM  ──────┤
                             │
Fold 2 ──┬── MLP    ─────────┤   각 모델의 fold별 확률을
         ├── XGBoost  ───────┼─→  평균낸 후 가중합 (soft voting)
         └── LightGBM  ──────┤    
                             │   ENSEMBLE_WEIGHTS:
... (Fold 5까지) ...         │     MLP 0.25 / XGB 0.45 / LGB 0.30
                             │
                             ↓
                       Final Prediction
```

### MLP 아키텍처

```
Input (50+ features)
  ├─ Linear(512) → BatchNorm → GELU → Dropout(0.3)
  ├─ Linear(256) → BatchNorm → GELU → Dropout(0.3)
  ├─ Linear(128) → BatchNorm → GELU → Dropout(0.3)
  ├─ Linear(64)  → BatchNorm → GELU           ← Residual: Input → 64로 skip + 더하기
  └─ Linear(3)
```

### 하이퍼파라미터

| 설정 | 값 | 이유 |
|---|---|---|
| Optimizer | AdamW (weight_decay=1e-4) | L2 정규화 내재화 |
| Learning Rate | 5e-4 | Cosine 스케줄러와 균형 |
| Scheduler | CosineAnnealingLR | 초반 탐색 + 후반 미세조정 |
| Batch Size | 256 | BatchNorm 통계 안정 + 속도 |
| Epochs | 100 (Early Stop) | Patience=15로 자동 종료 |
| Dropout | 0.3 | 과적합 방지 |
| Gradient Clip | max_norm=1.0 | 폭발 방지 |
| K-Fold | 5 (Stratified) | 분산 감소 + 앙상블 강화 |

<br>

## 📊 성능 결과

### 학습 곡선 (Best Fold)

![Training Curve](images/07_training_curve.png)

Cosine 스케줄러로 학습률이 점진적으로 감소하면서 후반부 val accuracy의 진동이 줄어드는 모습이 관찰됩니다. Early Stopping이 patience=15로 동작해 과적합 전 자동 종료됩니다.

### Confusion Matrix

![Confusion Matrix](images/08_confusion_matrix.png)

대각선(정답) 비율이 높고, Standard ↔ Good / Standard ↔ Poor 간 일부 혼동이 보입니다. 인접한 등급 사이의 경계는 본질적으로 모호한 영역이라 자연스러운 패턴.

### 모델별 성능 비교

| 모델 | Hold-out Test Accuracy |
|---|---|
| 기본 MLP (64→32→3, 베이스라인) | 71.52% |
| 개선 MLP + Residual | ~76% |
| XGBoost (5-fold 평균) | ~84% |
| LightGBM (5-fold 평균) | ~83% |
| **3-모델 가중 앙상블 (최종)** | **약 87~89%** |

### 출력 예시

```
============================================================
  K-Fold Cross Validation 평균 검증 점수
============================================================
  MLP: 75.21% (± 0.42%)
  XGB: 83.94% (± 0.31%)
  LGB: 82.87% (± 0.28%)

============================================================
  Hold-out Test 최종 평가
============================================================
  MLP only:          75.84%
  XGBoost only:      83.55%
  LightGBM only:     82.62%

  ★ 최종 앙상블 (soft): 87.93%
============================================================
  목표 기준 75% 달성
============================================================
```

<br>

## 💡 개선 포인트 요약

영향이 큰 순서대로 정리합니다.

| # | 개선 항목 | 기여도 |
|---|---|---|
| ① | **XGBoost + LightGBM 앙상블 도입** | +5~7%p |
| ② | **5-Fold Cross Validation** | +1~2%p |
| ③ | **Customer-level 집계 피처** (평균/표준편차 26개) | +2~3%p |
| ④ | **MLP + Residual 구조 개선** | +1.5%p |
| ⑤ | **Type_of_Loan Multi-hot Encoding** | +1%p |
| ⑥ | **학습 전략 개선** (AdamW + Cosine + Early Stop) | +1%p |
| ⑦ | **비율 피처 생성** (Debt/Income 등) | +0.5%p |
| ⑧ | **데이터 누수 차단** (fold 내부 scaler fit) | 누수 제거 |

<br>

## 🚀 실행 방법

### 1. 의존성 설치

```bash
pip install torch scikit-learn xgboost lightgbm pandas numpy matplotlib seaborn
```

### 2. 데이터 준비

`train2.csv`를 프로젝트 루트에 위치시킵니다.

### 3. 실행

```bash
python credit_score_competition.py
```

### Colab 사용 시

1. `런타임` → `런타임 유형 변경` → `T4 GPU` 선택
2. `train2.csv` 업로드
3. 코드 전체를 셀에 붙여넣고 실행

실행하면 `images/` 폴더에 EDA 시각화 6종 + 학습 곡선 + Confusion Matrix가 자동 생성됩니다.

### 실행 시간

| 환경 | 예상 시간 |
|---|---|
| Colab T4 GPU | 약 15~25분 |
| CPU (8 cores) | 약 60~90분 |

<br>

## ⚠️ 한계 및 향후 과제

- **약한 데이터 누수**: Customer-level 집계는 같은 고객의 행이 train/val에 섞이면서 약한 정보 누출 여지가 있음. 실제 서비스 시나리오가 "신규 고객 예측"이라면 **GroupKFold**(Customer_ID 기준)로 분할 필요.
- **시계열 활용**: 본 프로젝트는 8개월의 통계량만 사용. RNN/Transformer로 시계열 패턴 자체를 학습하면 추가 개선 가능.
- **앙상블 다양화**: CatBoost 추가, Stacking(MetaLearner) 도입 시 추가 향상 여지.
- **하이퍼파라미터 자동 탐색**: Optuna로 XGB/LGB의 max_depth, n_estimators 등을 자동 튜닝하면 단일 모델 성능 추가 향상 가능.

<br>

## 📁 파일 구조

```
.
├── credit_score_competition.py    # 메인 스크립트 (전처리 + EDA + 학습 + 앙상블)
├── train2.csv                     # 학습 데이터 (Kaggle에서 다운로드)
├── images/                        # 시각화 자동 생성 폴더
│   ├── 01_class_distribution.png
│   ├── 02_mutual_information.png
│   ├── 03_correlation_heatmap.png
│   ├── 04_features_vs_target.png
│   ├── 05_loan_type_distribution.png
│   ├── 06_customer_panel.png
│   ├── 07_training_curve.png      # 학습 후 자동 생성
│   └── 08_confusion_matrix.png    # 학습 후 자동 생성
└── README.md
```

<br>

## 📝 라이선스

학습 목적의 개인 프로젝트입니다. 데이터는 Kaggle 원본 라이선스를 따릅니다.
