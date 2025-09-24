# modulabs_projects
개인 파일럿 프로젝트 및 모두의연구소 Bootcamp 과정에서 진행한 다양한 데이터 분석 및 머신러닝 프로젝트들을 정리한 저장소입니다.  
각 서브 디렉터리는 날짜별/주제별로 구분되어 있으며, 실습한 주피터 노트북(`.ipynb`) 파일들이 포함되어 있습니다.

---

## 📂 Repository Structure

- **250728_eda**  
  - 데이터 탐색(EDA, Exploratory Data Analysis) 실습 노트북 모음  
  - 예시: 데이터 클렌징, 결측치 처리, 이상치 탐색, 시각화  

- **250729_data_transformation**  
  - 데이터 변환 및 전처리 관련 실습  
  - 스케일링, 인코딩, 정규화, 표준화 등의 작업 포함  

- **250730_feature_eng**, **250731_feature_eng**, **250801_feature_eng**  
  - 피처 엔지니어링(Feature Engineering) 관련 프로젝트  
  - 새로운 변수 생성, 파생 변수 추가, 도메인 지식 활용  

- **250821_main_quest_구매수량예측**  
  - 메인 퀘스트 프로젝트: **구매 수량 예측 모델**  
  - 다양한 고객 정보 데이터를 활용하여 고객이 구매한 물품 수량을 예측
  - Data pre-processing
  - Feature engineering
  - Model selection
  - Model training
  - model evaluation  

- **customer_segmentation_project**  
  - 고객 세분화 프로젝트  
  - 군집 분석, 고객 행동 데이터 기반 세그먼트 정의  

- **machine_learning_module**  
  - 머신러닝 모듈 학습 및 실습 프로젝트 모음  
  - 분류(Classification), 회귀(Regression), 클러스터링(Clustering) 등 다양한 모델 훈련 기록

- **250902_team_proj_residual_analysis**  
  - 선형 회귀 (Linear regression) 관련 프로젝트  
  - 잔차 분석 수행  
    - 잔차와 예측값 산점도를 통한 등분산성 검증  
    - 잔차 히스토그램 및 QQ 플롯을 통한 정규성 검증  
    - VIF(Variance Inflation Factor)를 통한 잔차 독립성 검정  
  - 목표: 잔차 분석을 기반으로 선형 모델 적용 가능성과 문제 해결 단서 탐색

- **250903_keggle_housing_price_pred**
    - Kaggle House Prices: Advanced Regression Techniques 실습 노트북
    - 주요 내용: 데이터 불러오기, 전처리, 탐색적 데이터 분석(EDA), 피처 엔지니어링, 회귀 모델 학습 및 평가
    - 목표: 건물과 관련된 여러가지 정보를 활용하여 주택의 가격 예측모델 학습/예측 및 평가

- **250904_clusturing**
    - Clusturing 학습내용 실습 코드
    - 주요 내용: K-means, Mean Shift, GMM, DBSCAN (군집화, 군집평가, 실습)
    - 목표: 주요 clusturing 알고리즘에 대한 이해 및 iris 데이터 세트를 이용한 실습.
    - 참고: https://www.kaggle.com/code/azminetoushikwasi/different-clustering-techniques-and-algorithms

- **250905_NLP**
    - 자연어처리
    - 텍스트 분석 / 감정(감성) 분석
    - 주요내용: NLP와 TA의 관계, 텍스트 전처리(Cleansing, Tokenization, Filtering/Stop Word Removal, Stemming/Lemmatization), 희소행렬
    - 목표1: 텍스트 마이닝 프로세스에 대한 이해 및 20 뉴스그룹 분류 데이터세트를 이용한 분류 모델 생성/학습/평가
    - 목표2: 감정분석을 통해 고객 후기 기반 서비스 호감도 예측 모델 생성/학습/평가

- **hom_3rd_practice**  
  - [O'REILLY] *Hands-On Machine Learning (3rd edition)* 교재 연습문제 실습 노트북 모음  
  - 각 장별로 제공된 예제 코드와 연습문제를 직접 구현하며 학습  
  - Notebook 파일명: `장_해당 장의 주제.ipynb` 형식으로 구성  
  - 목표: 교재 연습문제 풀이 및 실습을 통해 머신러닝, 딥러닝 개념 및 알고리즘 이해

- **time_series**
    - 시계열 분석관련 예제 및 학습 코드 정리
    - **250909_stationarity.ipynb**
        - 시계열의 정상성 관련 학습 코드
        - 주요내용:
            - 시계열 분석의 기본조건인 정상성의 정의
            - 시계열의 비정상 
            - 비정상 시계열의 정상화 방법
            - 정상성의 확인: KPSS(Kwiatkowski-Phillips-Schmidt-Shin Test) 검정 / ADF (Augmented Dickey-Fuller) 검정
            - 시각화를 통한 시계열 EDA
            - ACF(AutoCorrelation Function) Plot
            - PACF(Partial AutoCorrelation Function) Plot
        - 목표: 정상 시계열을 이해하고 차분, 평활 등을 통해 비정상 시계열을 정상화 하여 시각화 자료로 설명한다.
    - **250910_classification.ipynb**
        - 시계열 데이터를 활용한 분류(Classification) 문제 실습
            1. air passengers 데이터
               - 로그변환을 통한 분산 안정화
               - 차분을 통한 경향성 제거
            2. robot exection failures 데이터
               - 센서 id에 따른 stratified train_test set 분할
               - extract_features()
               - EfficientFCParameters()
               - RandomForest 활용 분류
               - XGBoost 활용 분류
    - **250911_01_ARIMA.ipynb**
        - Daily_Demand_Forecasting_Orders 활용 ARIMA 분석 실습
        - ACF/PACF - 시각화
        - ARIMA
        - AutoARIMA
    - **250911_02_ARIMA_ARCH_miniproject.ipynb**
        - 금융 시계열 데이터에 대한 분석 방법들
            - ARCH
            - GARCH
        - Air Passengers 데이터를 활용한 ARIMA
        - S&P 500 데이터셋을 활용한 ARCH
    - **250912_time_series_full.ipynb**
        - 시계열 데이터를 다루는 전 과정 복기 및 ARIMA 분석까지
        - 시계열 데이터 전처리 과정 및 시계열 데이터의 요소들에 대한 이론정리
    -**250915_time_series_final.ipynb**
        -  업비트 코인거래 차트를 활용한 추세 예측 프로젝트
        -  목표: 상승장과 하락장을 예측해본다
        -  주요 내용: Data labeling, Feature engineering, Featur selection, Classifier 훈련/예측/평가
      **deep_learning**
        - 딥러닝 학습과정 기록
        - 딥러닝 모델 활용 프로젝트 모음 (이미지분류 등)
          
---

```bash
git clone https://github.com/milkpotato1000/modulabs_projects.git
```
를 통해 저장소를 클론할 수 있습니다.

---

## 🎯 Purpose

- 데이터 분석 및 머신러닝 학습 과정 기록

- Bootcamp 커리큘럼 복습 및 과제를 통한 실전 경험 축적

- 프로젝트 기반 포트폴리오 아카이브

---

## 📌 Note

각 프로젝트 노트북은 독립적으로 실행 가능하도록 작성되었습니다.