# 문제 해결 및 개선점

1. 다른 로컬 환경에서 git clone 만을 했을 때, 전체 pipeline 이 실행되도록 만들기
* mlflow 를 이용해 run의 중간 파일들을 artifacts 에 log 한다.
    * 이 때, run id 를 임시 텍스트 파일로 저장한다
* 이후의 run 에서 중간 파일들을 사용하고자 할 때:
    * 해당 파일을 artifact 로 가지고 있는 run 의 id 조회
    * run id 를 통해 artifact 에 접근하고 파일을 load
* 향후 개선점
    * Cloud Storage 이용

2. sample train data 생성 시 target label 에 대해 stratified sampling 을 진행했는데, 이 때 feature 에서 발생하는 bias 에 대한 처리?
=> 질문에 제대로 된 답변을 하지 못했었다. 그래서 현업자분께 여쭤봤다
* 우선 데이터 크기가 크지 않고 label ratio 도 균등했기 때문에 우선 원본 데이터로 실험 진행
=> 이후에 성능이 잘 나오지 않았을 때 이와 같은 방법을 고려하는 순서가 일반적이다
* 이런 질문을 받았을 때 어떤 처리를 했는지 설명할 수 없다면 결과로 보여주면 된다
=> 결국 model 성능 지표가 잘 나왔으니 신뢰할만 하다
* 성능이 잘 나오지 않아 undersampling 등이 필요하다면?
    * sampling 비율을 늘려 bias 를 줄이자
    * multivariate sampling 을 이용해 비율을 고려할 수 있는 key feature 까지 stratify
    * sampling 에 iteration 을 두어, 가장 성능이 좋은 것을 채택
    * sampling 후 feature 의 분포를 시각화해 bias 가 적다는 것을 보여주는 것도 방법

3. model metrics 선정
=> 나는 ml model evaluation 에 사용되는 대표적인 지표 (accuracy, f1, ROC AUC, recall, precision) 을 사용했다.
* 그러나, project 의 domain 에 따라 현업에서 실제로 사용되는 metrics 가 다르다.
* 이를 조사해보는 것이 필요하다
* ex: 대출심사 분류 모델 평가에서 GINI 계수가 주로 사용됨
=> 기본적으로 tree model 에서 사용되는 지표인데, 다른 binary classification model 에서도 사용하는 방법이 있다.
* 또한, ternary classification 에서 간과한 점:
    * ROC AUC, recall 등의 지표는 binary classification 에서 의미가 큰 것
    * One vs Rest 등의 방법을 사용해 metric 측정 방법을 수정해보는 것도 필요하다

4. classification task 설정
* 우리 프로젝트는 미연체 / 소액연체 / 거액연체 의 ternary classification 을 진행
* multilabel classification 보다는 여러 개의 binary classification 으로 설정하는 것도 고려해보자
    * 미연체 / 연체 분류 후 연체자를 소액 / 거액으로 분류