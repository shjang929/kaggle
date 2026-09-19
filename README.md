# 생존 예측 — Kaggle InClass 비공개 대회 (SKN-36) 2위 기록

Kaggle InClass 비공개 대회 [SKN-36](https://www.kaggle.com/competitions/skn-36) (주최: 조경원, 참가자 16명)의 생존 예측 과제에서 2위를 기록한 프로젝트입니다.
초반에는 검증 방법론 자체의 버그로 존재하지 않는 고득점에 속고 있었고, 이를 바로잡은 뒤
피처 엔지니어링과 하이퍼파라미터 튜닝을 반복하며 점진적으로 성능을 개선했습니다.

시행착오 전체 과정과 최종 파이프라인에 대한 자세한 설명은
네이버 블로그(https://blog.naver.com/leeran929/224413967486)에 정리되어 있습니다.

## 최종 결과

| 지표 | 값 |
|---|---|
| 모델 | CatBoostClassifier |
| 5-fold CV Accuracy | 0.8712 |
| 5-fold CV AUC | 0.9107 |
| 최종 순위 | 2 / 16 (Private Leaderboard) |

## 프로젝트 구조

```
.
├── data/                        # train.csv, test.csv, submission.csv (미포함, 아래 참고)
├── common/                      # 공용 모듈
│   ├── modeling.py              #   Modeling 클래스 (CatBoost/XGBoost/LightGBM 학습·평가·최적 모델 선택)
│   ├── evaluations.py           #   평가 지표 유틸
│   ├── evaluation_plots.py      #   혼동행렬 등 시각화
│   └── utils.py                 #   시드 고정 등 공용 유틸
├── submission/                  # 제출 파일 (미포함, 노트북 실행 시 재생성)
├── initial.ipynb                # 초기 EDA / 피처 탐색
├── 1~8. base model*.ipynb       # 1차 탐색 (SMOTE, 피처, CV, AutoML, 베이지안 HPO, SHAP)
├── 9. score_history.ipynb       # 전체 노트북 점수 비교 시각화
├── 10~15. final_0*.ipynb        # 홀드아웃 재설계 이후 신뢰 가능한 개선 시리즈 (최종 제출작: 15번)
├── titanic_blog_post.md         # 전체 시행착오 기록 (블로그 원고)
├── score_history_*.png          # 위 9번 노트북이 생성하는 비교 그래프
├── pyproject.toml / uv.lock     # 의존성 정의 (uv)
└── .python-version              # Python 3.12
```

## 노트북 진행 순서

| 번호 | 파일 | 내용 | 비고 |
|---|---|---|---|
| 1 | base model | 베이스라인 | 검증 버그로 신뢰 불가 |
| 2 | base model_SMOTE | 오버샘플링 | 유일하게 별도 검증 분할 있음 |
| 3 | base model_feature | 피처 추가 | 검증 버그로 신뢰 불가 |
| 4 | base model_feature_cv | 5-fold CV 도입 | 검증 버그로 신뢰 불가 |
| 5 | base model_feature_autoML | AutoML(pycaret) | 검증 버그로 신뢰 불가 |
| 6 | base model_feature_Bayesian | Optuna(TPE) 베이지안 HPO | 검증 버그로 신뢰 불가 |
| 7 | base model_feature_HPO | 이름과 달리 튜닝 코드 없음(3번과 동일 구조) | 검증 버그로 신뢰 불가 |
| 8 | base model_feature_shap | SHAP 피처 중요도 분석 | 검증 버그로 신뢰 불가 |
| 9 | score_history | 1\~15번 점수 종합 비교 시각화 | — |
| 10 | final_01 | 홀드아웃 검증 재설계 (여기부터 신뢰 가능) | Acc 0.8700 / AUC 0.9060 |
| 11 | final_02 | 신규 피처 7개 일괄 추가 | 회귀, Acc 0.8646 / AUC 0.9040 |
| 12 | final_03 | 피처 원복 + Optuna 멀티오브젝티브 튜닝 시작 | Acc 0.8700 / AUC 0.9085 |
| 13 | final_04 | `is_married_woman` 피처 추가 | Acc 0.8700 / AUC 0.9092 |
| 14 | final_05 | 하이퍼파라미터 탐색 범위 확장 | Acc 0.8722 / AUC 0.9073 (트레이드오프) |
| 15 | **final_06** | **가족/그룹 생존율 피처 추가 (최종 제출작)** | **Acc 0.8712 / AUC 0.9107** |

1~8번은 검증(`test_score`) 계산에 학습 데이터를 그대로 재사용하는 버그가 있어 자체 출력 점수를
신뢰할 수 없습니다. 10번부터 홀드아웃 분리와 5-fold CV를 정상적으로 적용해, 이후 모든 비교는
10\~15번 값을 기준으로 합니다. 자세한 내용은 블로그 원고를 참고하세요.

## 핵심 방법론

- **검증**: `train_test_split` 홀드아웃(20%) + `StratifiedKFold` 5-fold CV 병행
- **하이퍼파라미터 튜닝**: Optuna 멀티오브젝티브 최적화(`directions=['maximize', 'maximize']`,
  accuracy·AUC 동시 목적함수) — accuracy가 기준값 이상인 후보 중 AUC가 최대인 지점 선택
- **가족/그룹 생존율 피처**: 티켓 번호·성(姓)·요금으로 동승 그룹을 식별하고, 그룹 내 생존율을
  leave-one-out 방식으로 계산(홀드아웃/CV 각 fold/최종 학습 세 시점 모두 누수 없이 별도 계산)
 
## 실행 방법

```bash
# 의존성 설치 (uv)
uv sync

# Jupyter 실행
uv run jupyter lab
```

`data/` 폴더에 `train.csv`, `test.csv`, `submission.csv`를 넣은 뒤 노트북을 순서대로 실행하면
됩니다. 데이터 파일은 대회 규정상 리포지토리에 포함하지 않았습니다.

## 기술 스택

Python 3.12 · CatBoost · XGBoost · LightGBM · scikit-learn · Optuna · pandas · SHAP · uv

## 대회 정보 / 인용

이 프로젝트는 Kaggle InClass 비공개 대회 SKN-36을 위해 진행되었습니다.

```bibtex
@misc{skn-36,
    author = {조경원},
    title = {SKN-36},
    year = {2026},
    howpublished = {\url{https://www.kaggle.com/competitions/skn-36}},
    note = {Kaggle}
}
```
