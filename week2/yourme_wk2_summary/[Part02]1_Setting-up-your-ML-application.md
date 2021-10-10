# 1. Setting up your ML application

### Train / dev / test sets

![Untitled](1%20Setting%20up%20your%20ML%20application%20b86a35ee91a945da81cc685930acf3c8/Untitled.png)

1. Idea : 특정 개수의 층과 유닛을 가지고 특정 데이터 세트에 맞는 신경망을 만듦
2. Code : 코드로 작성하고 실행하고 실험 진행
3. Experiment : 특정 네트워크 혹은 설정이 얼마나 잘 작동하는지 알게 됨
    
    ⇒ 결과에 기반하여 아이디어를 개선하고 몇 가지 선택을 바꾸게 됨
    

⇒ 더 나은 신경망을 찾기위해 이 과정을 반복함

- 빠른 진전을 이루는 데 영향을 미치는 것
    - 사이클을 얼마나 효율적으로 돌 수 있는지
    - 데이터 세트를 잘 설정하는 것
- Train Set
    - 훈련을 위해 사용되는 데이터
    - 전통적인 방법
        - Data set = traing set + Hold-out cross validation set(= Development set ⇒ "dev set") + test set
    - Pre ML : 70(train)/30(test), 60(train)/20(dev)/20(test)
    - Big Data Era : dev & test 세트가 훨씬 더 작아지는 것이 트렌드 (98/1/1)
        
        $\because$  dev set & test set 목적 : 서로 다른 알고리즘을 시험하고 어떤 알고리즘이 더 작동하는지 확인하는 것
        
        ⇒ dev set : 평가할 수 있을 정도로만 크면 됨.
        
- Dev Set
    - 다양한 모델 중 어떤 모델이 성능을 나타내는지 확인함.
- Test Set
    - 목표 : 최종 네트워크의 성능에 대한 비편향 추정을 제공하는 것
    - 비편향(unbiased) 추정이 필요 없는 경우에 테스트 세트를 갖지 않아도 됨

### Bias/Variance

- 편향-분산 트레이드 오프
    
    ![Untitled](1%20Setting%20up%20your%20ML%20application%20b86a35ee91a945da81cc685930acf3c8/Untitled%201.png)
    
    ![Untitled](1%20Setting%20up%20your%20ML%20application%20b86a35ee91a945da81cc685930acf3c8/Untitled%202.png)
    
- 훈련 세트와 개발 세트의 관계
    
    ![Untitled](1%20Setting%20up%20your%20ML%20application%20b86a35ee91a945da81cc685930acf3c8/Untitled%203.png)
    
    - 가정: 인간은 대략 0%의 오차를 냄(인간 수준의 성능이 거의 0%)
        
                  = 베이지안 최적 오차가 0%
        
    - 훈련 세트 오차를 확인 : 편향 문제가 있는지 알 수 있음.
    - 훈련세트 → 개발세트 : 오차가 얼마나 커지는지에 따라서 분산 문제를 알 수 있음.
    
     ⇒  가정: 베이즈오차가 꽤 작고, 훈련 세트와 개발 세트가 같은 확률 분포에서 왔다
    

### Basic "recipe" for machine learning

![Untitled](1%20Setting%20up%20your%20ML%20application%20b86a35ee91a945da81cc685930acf3c8/Untitled%204.png)