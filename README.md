# README

#### 개발 환경
- Tensorflow ver 1.12
- scikit-learn 0.19.1
- Anaconda 5.2.0
- xgboost 0.81 (GPU accelerated version)
- seaborn 0.9.0

#### 첨부 파일 목록
- CAM_CNN.py 
- CNN_module.py
- XGBoost_GPU.py
- EDA.ipynb

#### 실행 방법

EDA.ipynb는 시각화 및 doc2vec 코드가 포함되어 있습니다.

```python
python XGBoost_GPU.py 
python CAM_CNN.py 
```

XGBoost.py를 실행시 학습 진행 및 결과를 추출합니다. 다만 XGboost GPU version이 설치되어 있어야 합니다.  

CAM_CNN.py를 실행하면 graph와 result 폴더가 생성됩니다. result 폴더에는 다음과 같은 값들이 저장됩니다.

- result.csv : test data에 대한 cam score 
- star_results.csv : 상위 5개 ingredient를 별로 표시한 데이터
- top_dict.json : 각 cuisine 상위 ingredient count dictionary

graph 폴더에는 모델과 텐서플로우를 통해 추출한 summary 값들(loss와 accuracy)가 저장됩니다. 
또한 embedding 된 값들에 PCA와 t-sne 를 적용할 수 있습니다. 텐서보드 명령어는 다음과 같습니다.

```python
tensorboard --logdir="./graph/"
```


모델의 전체적인 모양은 다음과 같습니다.

![graph](./assets/graph.png)

