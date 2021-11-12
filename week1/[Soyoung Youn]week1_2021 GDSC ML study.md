# GDSC machine learning study

# 1. 딥러닝 소개

## Deep Learning

: 인터넷 산업(Ex) 웹 검색, 광고...)에 이미 사용되고 있음. 새로운 종류의 상품이나 산업의 개발도 가능하게 함. x-ray 이미지 분석부터 헬스케어 분야, 개인 맞춤형 교육, 정교한 농업기술, 자율주행 자동차까지 다양한 것들을 창조해 내고 있다.

: Ai is the new Electricity. 100년 전 전기화는 많은 변화(통신, 헬스케어, 제조, 운송 등)를 가져옴. 오늘날에는 인공지능이 그것을 보여주고 있음

: 이러한 급격하고 빠른 변화를 이끌어내는 인공지능의 한 부분은 딥러닝이다. 딥러닝은 과학기술사회에서 가장 필요한 능력 중 하나가 되었다.

### What is a Neural Network?

딥러닝이라는 용어는 신경망을 학습시키는 것을 칭한다. 종종 아주 큰 신경망을 학습시키기도 한다. 그렇다면 신경망이란 정확히 무엇일까?

: 신경망이란 입력(x)과 출력(y)을 매칭해주는 함수를 찾는 과정이다. 

- Housing Price Prediction(주택 가격 예측) 예제로 신경망 개념 이해하기
    
    > 6개의 주택 데이터를 가지고 있다. 이때, 제곱 피트나 평방 미터로 되어 있는 주택의 크기를 알고 있고 주택의 가격을 알고 있을 때 주택의 가격을 예측할 수 있는 함수를 만든다고 해보자
    > 
    - 선형회귀를 잘 알고 있다면 아마 x축이 size of house이고 y축이 price인 함수에서 6개의 주택 데이터가 찍힌 부분에 직선을 그리자고 할 수도 있을 것이다.
    - 좀 더 나은 함수 → 주택 가격은 음수가 될 수 없다는 사실을 이용한다. 즉, 음수 부분에 닿을 직선 대신에 선을 꺾어서 그려 0으로 끝나도록 하는 그래프가 그려지는 것이다.
    
    **⇒ 이제 주택 가격을 예측하는 함수를 신경망으로 생각해보자.** 
    
    1. X라고 불릴 주택의 크기가 신경망의 입력이 된다. 
    2. 입력이 작은 원인 '노드'로 들어가게 된다.
    3. Y라고 불릴 주택의 가격을 출력하게 된다. 
    - 여기서 작은 원(="노드")이 신경망에서 하나의 뉴런이 된다. 이 뉴런이 주택 가격을 예측하는 함수를 구현하게 되는 것이다.
    
    > 뉴런이 하는 일은 주택의 크기를 입력으로 받아서 선형 함수를 계산하고 결과 값과 0 중 큰 값을 주택의 가격으로 예측하는 것이다.
    > 
    - 신경망 논문에서는 현재 예제에서 본 것과 같은 0에서 시작하여 직선으로 연결되는 형식의 "ReLu(Rectified Linear Unit)"함수가 많이 나타난다. Rectified는 결과값과 0 중 더 큰 값을 취하라는  뜻이다.

위 예제에서 살펴본 신경망은 하나의 뉴런으로 되어 있다. 하지만, 더 큰 신경망은 더 많은 뉴런들을 쌓아놓은 모양으로 존재한다. (뉴런 하나를 하나의 레고 블럭으로 생각해도 좋다. 많은 레고 블럭들을 쌓음으로써 더 큰 신경망을 만들 수 있다.)

- Housing Price Prediction(주택 가격 예측) 예제로 더 큰 신경망 이해하기
    
    > 주택의 크기에서 주택의 가격을 예측하는 대신, 다른 특성(침실 수 등)이 있다고 생각해보자. 주택의 가격에 정말로 미치는 것은 가족의 크기다. 가족의 명수에 따라 피트 제곱, 평방 미터, 침실 수 등은 가족의 크기에 적당한지 다르게 판단된다. 우편번호(zip code/postal code)는 걷기에 좋은 동네인지를 알려주는 특성일 수도 있다. 우편번호와 부는 명문 학교의 존재, 즉, 학군을 알려주는 척도가 될 수 있다.
    > 
    - size, bedrooms 특성을 고려한 하나의 뉴런, zipcode 특성을 고려한 또 하나의 뉴런, 그리고 zip code와 wealth라는 특성을 고려한 또 다른 하나의 뉴런은 ReLu 함수가 될 수 있고, 비선형 함수가 될 수 있다.
    - 현재 예제에서 X는 4개의 입력이고 Y는 예측하고자 하는 주택의 가격이다.
    - 뉴런이나 간단한 예측기들을 쌓아올림으로써 첫 번째 예제보다 더 큰 신경망을 가지게 되었다.
    
    ⇒ 그렇다면 이런 신경망의 마법은 어떻게 발생했을까? 입력과 출력 중앙에 있는 것들은 알아서 스스로 알아내는 것일까?
    
    : 중간에 있는 원들은 "신경망의 은닉 유닛"이라고 불린다. 이 유닛 각각은 4개의 입력을 받는다.
    
    - 예를 들어, 중앙에 있는 신경망의 은닉 유닛 중 하나가 size를 내포한다고 하자. 이때, 단순히 size에 대한 입력만 받아들이지 않고, 4개의 입력 모두를 받아들인다. 입력층과 중앙에 존재하는 층은 조밀하게 연결되어 있다고 이야기할 수 있다.
    
    > 즉, 모든 입력 특성들은 중앙에 있는 원 모두에 연결되어 있다.
    > 
    - 뉴런이 모든 입력을 받아들인다고 해서 관계 여부에 대해 걱정할 필요는 없다. 관계 여부는 신경망이 학습하면서 알아서 조절해 주기 때문이다.

**[정리]**

: 충분한 양의 (x, y)를 훈련 예제로 제공한다면, 신경망은 x를 y로 연결하는 함수를 알아내는 데 정말 뛰어날 것이다.

→ 내가 생각하고 이해한 것: 어떤 입력 x를 출력 y에 연결하는 일을 하고 싶을 때 신경망을 활용하자! 단, 인간은 신경망이 학습할 수 있는 충분한 양의 데이터를 제공해야 한다.

---

### Supervised Learning with a Neural Network

**[머신러닝의 방법 2가지]**: 물론 실제 방법은 2가지보다 많다.

1. 지도 학습: 정답이 주어져있는 데이터를 사용하여 컴퓨터를 학습시키는 방법
    
    → what is neural network에서 배운 신경망을 통해 통해 지도 학습 구현할 수 있다.
    
    ![스크린샷 2021-10-01 오후 11.39.18.png](GDSC%20machine%20learning%20study%20e20bbf21ed934a95bdecd5f9eb2e8890/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2021-10-01_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_11.39.18.png)
    
2. 비지도 학습

**지도 학습 : 입력 X와 출력 Y에 매핑되는 함수를 학습하려고 함**

[예시](https://www.notion.so/3d5a96483cc9425f8509b1a83c8f0064)

cf) 오디오는 시간의 흐름에 따라 재생되며, 주로 1차원의 시계열 데이터로 나타나는 시퀀스 데이터이다.

> 신경망을 통한 많은 값들의 생성은 어떤 문제를 해결하기 위한 적절한 X와 Y를 통해 이루어지고 자율주행 같은 더 큰 시스템에 지도학습 요소들이 적합하게 한다. 또한, 조금씩 다른 신경망들은 서로 다른 각자의 적절한 응용분야에 적용된다.
> 

**[신경망 특징]**

- 분야에 따라 적용되는 신경망이 다름

       ex) 이미지 분류를 위해 CNN 사용, 음성을 텍스트로 변환시키기 위해 RNN 사용

- 구조적 및 비구조적 데이터를 신경망을 사용하여 예측할 수 있다.

**[Neural Network Examples]**

1. Standard NN(표준 신경망)
2. Convolutional NN(합성곱 신경망)
    - 이미지 데이터는 보통 CNN에서 사용된다.
3. Recurrent NN(순환 신경망)
    - 1차원 시퀀스 데이터에 강하다. ← 시간적 요소가 시퀀스 데이터에 자주 포함된다.

**[신경망에 사용되는 데이터]**

1. 구조적 데이터(Structure data)
    
    : 기본적으로 데이터베이스로 표현된 데이터 
    
    → 정보의 특성이 잘 정의되어 있다.
    
2. 비구조적 데이터(Unstructure data)
    
    : 이미지, 오디오, 텍스트와 같이 특징적인 값을 추출하기 어려운 형태의 데이터
    
    → 여기서의 특성은 이미지의 픽셀값, 텍스트의 각 단어 같은 것이다.
    
    → 딥러닝/신경망 덕분에 컴퓨터가 비구조적 데이터를 인식할 수 있게 되었다.
    
- 구조적 데이터보다 비구조적 데이터가 컴퓨터가 작업하기 어렵다.
- 신경망에서 발생하는 많은 경제적 이익은 광고 시스템이나 사용자 맞춤 추천 등의 구조적 데이터에서 오는 경우가 많다.

---

### Why is deep learning taking off?

1. 데이터 양 증가(Data)
    - 과거와 현재 존재하는 데이터 양의 차이
        
        > 과거에는 많은 문제에 대해 상대적으로 적은 양의 데이터가 존재했지만, 현재에는 꽤 많은 양의 데이터를 보유하게 되었다.
        > 
        1. 디지털 기기 사용의 증가
            1. Ex) 웹 사이트, 모바일 앱 등
            2. 디지털 기기 상의 활동은 데이터를 생성함
        2. 휴대폰 안 저가 카메라나 가속도계, IOT에 사용되는 수많은 센서 등의 증가
        3. 계속해서 데이터를 수집하는 노력
        
    - Amount of data / Labeled data, Performance 그래프와 신경망 사이의 관계
        
        > X축이 amount of data / Labeled data, Y축이 Perfomance일 때, 그래프는 원점에서 증가하다가 일정 데이터 양이 되면 수평선이 되는 함수가 존재한다. *레이블 데이터: 입력값 x와 레이블 y가 같이 있는 훈련 세트를 말함(강좌에서 m = 훈련 세트의 크기 → x축에 사용됨)
        > 
        
        ⇒ 이때, 신경망을 활용하면 어떻게 될까?
        
        : 위와 같은 상황에서 신경망은 더 좋은 Performance를 구현시키는 데 이바지할 수 있다. 
        
        - 크기가 클수록 더 좋은 Performance로 나타난다.
        - m이 작은, 즉, 훈련 세트의 크기가 작은 쪽(그래프의 왼쪽)에서는 알고리즘의 상대적 순위가 잘 정의되어 있지 않으므로 데이터 특성을 다루는 실력이나 알고리즘의 작은 부분이 성능을 크게 좌우한다.
        - m이 큰, 즉, 훈련 세트의 크기가 큰 쪽(그래프의 오른쪽)에서는 큰 신경망이 일관되게 다른 방법을 압도하는 경향을 보인다.
2. 컴퓨터 성능 향상(Computation: 계산 능력)
    
    : CPU, GPU에서 아주 큰 신경망을 훈련시키게 된 것 → 큰 성과를 낼 수 있었음
    
    - 빠른 계산의 중요성
        1. 빠른 계산이 중요한 이유: 대부분의 경우 신경망을 학습시키는 과정이 '반복적'임
            - 과정: Idea → code 0 → Experiment
            - 이 과정의 반복은 신경망의 훈련 시간, 속도와도 관련되어 있다.
            
            ⇒ 빠른 계산 방법은 실험의 결과를 얻는 시간을 더 빠르게 만들었고, 이는 신경망을 사용하는 사람과 딥러닝을 연구하는 사람 모두에게 긍정적 영향을 끼쳤다. 과정을 빨리 반복해 아이디어를 빨리 발전시킬 수 있도록 했기 때문이다.
            
        2. 빠른 계산 방법은 새로운 알고리즘 발명하는 데 큰 도움을 줌
3. 알고리즘(Algorithms)
    - 알고리즘의 발전은 많은 데이터를 가진 큰 신경망을 적당한 시간 내에 처리하거나 더 큰 신경망을 훈련할 수 있게 한다.
    - 알고리즘 혁신의 많은 부분이 신경망을 더 빠르게 실행하는 데 관한 것이었다.
    - sigmoid 함수와 ReLU 함수
        
        ![스크린샷 2021-09-27 오전 2.15.16.png](GDSC%20machine%20learning%20study%20e20bbf21ed934a95bdecd5f9eb2e8890/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2021-09-27_%E1%84%8B%E1%85%A9%E1%84%8C%E1%85%A5%E1%86%AB_2.15.16.png)
        
        **Sigmoid Function**
        
        : 왼쪽, 오른쪽 끝으로 가면 미분값이 0이 되기 때문에 Gradient(기울기)가 거의 0이 되며 소멸한다.
        
        이때, 함수의 경사가 0인 곳, 즉, 함수의 기울기가 0이 되면 학습이 굉장히 느려지는 문제가 발생한다. 
        
        따라서, 머신러닝에서 sigmoid 함수를 사용하는 데에는 문제가 있다.
        
        **ReLU Function**
        
        : 신경망의 활성화 함수를 ReLU 함수로 함으로써 Sigmoid 함수의 문제를 해결할 수 있다.
        
        - ReLU 함수는 입력값이 양수인 경우 경사가 1로 모두 같다.
            
            → 경사가 서서히 0에 수렴할 가능성이 적다.(=학습이 느려질 가능성이 적다.)
            
        - 물론, ReLU 그래프의 왼쪽은 0이다.
        
        ⇒ 그래도 ReLU 함수를 Sigmoid 함수 대신 사용함으로써 경사 하강법이라는 알고리즘을 훨씬 빠르게 만들었다. 이 알고리즘은 계산 능력(Computation)을 크게 향상시켰다.
        
    - 함수의 기울기와 학습 간의 상관관계
        
        함수 그래프의 기울기, 즉, 경사가 0일 때 '경사 하강법'을 사용하면 파라마터(변수)가 아주 천천히 바뀐다. 이때, 학습 속도도 느려진다.
        

# 2. 이진 분류(Binary Classification)

### 이진 분류(Binary Classification)

: 그렇다, 나이다 2개로 결과를 분류하는 것이다. 이때, 결과가 '그렇다'이면 1로 표현하고, '아니다'이면 0으로 표현한다.

- 이진 분류 문제의 예
    
    > input = image → cat(1) vs non cat(0)
    > 
    
    ⇒ 입력된 사진의 크기(픽셀 기준)와 크기가 같고, 각 색에 대한 강도가 숫자로 적힌 RGB 3개의 행렬을 가지고 특성 벡터 x 한 열을 만든다. 이때, 특성 벡터 x는 R, G, B 세 개의 행렬에 적힌 모든 원소를 하나의 열로 적는다. 따라서 특성 벡터 x의 차원 n_x(n)은 64 * 64 * 3, 즉, 12288이다. 이 특성 벡터 x를 가지고 y값, 즉, 고양이인지 아닌지에 따라 1, 0 중 하나를 출력할 수 있도록 예측하는 분류기를 만드는 것이다.
    
- 자주 쓰이는 표기법
    1. 훈련 샘플의 입력과 출력 → (x, y)
    2. 레이블 → y ∈ {0, 1}
    3. m개의 훈련 샘플을 포함하는 훈련 세트 → m training example  : {(x^1, y^1), (x^2, y^2), ... , (x^m, y^m)}
    4. m(m_train, m_test, ...)
    5. x = [[..., x^1, ...], [..., x^2, ...], ..., [..., x^m, ...] ] : 세로 = n_x, 가로 = m
    6. x ∈ R^(n_x * m)
    7. 파이썬 행렬 크기 알 수 있는 방법 → 행렬이름.shape : Ex) X.shape = (n_x, m)
    8. Y = [y^1, y^2, ..., y^m]
    9. Y ∈ R^(1 * m)
    10. 파이썬에서 Y의 차원 알 수 있는 방법 → Y.shape : Ex) Y.shape = (1, m)
    
    ⇒ 나중에 x, y 외에도 여러 훈련 샘플에 관련된 데이터를 각각의 "열"로 놓는 것이 좋을 것!
    
    ![스크린샷 2021-10-01 오후 11.39.05.png](GDSC%20machine%20learning%20study%20e20bbf21ed934a95bdecd5f9eb2e8890/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2021-10-01_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_11.39.05.png)
    
- 신경망을 학습하는 계산 과정
    1. 정방향 패스(정방향 전파)
    2. 역방향 패스(역전파)

---

### 로지스틱 회귀(Logistic Classification)

: 이진 분류를 하기 위해 사용되는 알고리즘 → 나중에 더 자세히 배울 예정

- 로지스틱 회귀
    
    정의 : 답이 0 또는 1로 정해져 있는 이진 분류 문제에 사용되는 알고리즘
    
    추가 설명(검색) : 로지스틱 회귀는 "사건의 발생 가능성을 예측"하는 데 사용되는 통계기법으로, 목표는 일반적인 회귀분석과 같게, 독립변수와 종속변수 간의 관계를 구체적인 함수로 나타내어 향후 예측 모델에 사용하는 것이다.
    
    - X(입력 특성), y(주어진 입력 특성 X에 해당하는 실제 값),  ^Y(y의 예측값) = P(y = 1 | x)
    - 로지스틱 회귀 변수 → w: n_x 차원의 벡터 → w ∈ R^(n_x), b: 실수 → b ∈ R
    - ^y = S(W^T_x + b)  ←  (W^T_x: W의 전치)
    - "전치" : 전치 행렬 → m * n 행렬의 전치 행렬 = n * m의 행렬
    
    → 변수 w, b 두 개를 사용하지 않고 하나만 사용할 수도 있음
    
    - x_0 = 1, x ∈ R^(n_x + 1), ^y = S((θ^T_(n_x)), θ = [θ_0, θ_1, ..., θ_(n_x)] ← (θ_0 = b, θ_1~θ_(n_x) = w)
    
    [시그모이드 함수]
    
    - 그래프
    
    ![스크린샷 2021-10-01 오후 11.19.41.png](GDSC%20machine%20learning%20study%20e20bbf21ed934a95bdecd5f9eb2e8890/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2021-10-01_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_11.19.41.png)
    
    - 식
    
    ![스크린샷 2021-10-01 오후 11.12.04.png](GDSC%20machine%20learning%20study%20e20bbf21ed934a95bdecd5f9eb2e8890/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2021-10-01_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_11.12.04.png)
    
    → If x large positive number : S(x) ≈ 1 / (1 + 0^+) ≈ 1
    
    → If x large negative number : S(x) ≈ 1 / (1 + ∞) ≈ 0
    

⇒ 그럼 ^y와 y의 비교를 통해 결과 예측이 얼마나 잘 되었는지 확인하는 과정이 필요 → 손실 함수와 비용 함수가 활용된다.

---

### 비용 함수(Cost Function)

: 모든 입력에 대한 오차를 계산하는 함수로, 모든 입력에 대해 계싼한 손실 함수의 평균값으로 구할 수 있다.

### 손실 함수(Loss Function)

 : 하나의 입력 특성(x)에 대한 실제값(y)와 예측값(^y) 사이의 오차를 계산하는 함수이다.

→ 왜 이런 함수를 사용하나요?

: 우리의 목표는 "오차를 최소화하여 실제값에 가까운 예측값을 도출하는 알고리즘, 함수를 구현하여 신경망을 구축하는 것"이기 때문이다.

- 손실 함수
    
    [방법 1] : 로지스틱 회귀에서는 주로 사용하지 않는다.
    
    → 이유: 함수의 매개변수를 학습하기 위해 풀어야 할 최적화 함수가 볼록하지 않기 때문이다. 최적화 함수가 볼록하지 않다 = 여러 개의 지역 최적값을 가진다 = "예측하고 가능성을 따지는 데 있어서 여러 개의 최적값을 가질 경우 경사 하강법이 전역 최적값 찾지 못할 수 있음" ⇒ 경사 하강법 활용하지 못함
    
    L(^y, y) = 1/2 * (^y - y)^2
    
    [방법 2] : 방법 1의 함수와 비슷하지만 최적화 함수가 볼록해지는 또 다른 함수를 활용하자
    
    L(^y, y) = - (y * log (^y) + (1 - y) * log (1 - ^y))
    
    ⇒ 이 함수를 사용하면서 우리가 해야 할 것: 나온 오차를 줄이는 것
    
- 비용 함수
    
    : m개의 훈련 샘플 각각에 대한 손실 함수의 값을 전부 더해 그것을 훈련 샘플 총 개수인 m으로 나눈 것
    
    J(w, b) = 1 / m * Σi=1~i=m L((^y)^(i), y^(i))
    

⇒ 로지스틱 회귀 모델을 학습하는 궁극적 목표 = 비용 함수 J를 최소화하는 매개 변수 w와 b를 찾는 것 → 이를 위해서는 경사 하강법이 활용된다.

---

### 경사 하강법(Gradient Descent)

: 가장 빠르게 내려올 수 있는, 즉, 기울기가 가장 가파른 방향을 따라서 최적의 값으로 한 단계씩 내려오면서 점차 J(w, b)가 최소인 부분으로 도달한다.

→ 함수의 최솟값을 찾는 것이 목표인데, 이때 임의의 점을 골라 거기서부터 시작해 최적의 값, 즉, 최솟값을 가지는 w와 b를 찾아나간다. (= 임의의 값으로 초기화한다.)

- 보통 로지스틱 회귀에는 무작위 초기화 사용하지 않는다 → 0으로 초기화하는 경우가 대부분임

→ Repeat    {

w := w - a * (dJ(w, b) / dw)

b := b - a * (dJ(w, b) / db)

}

- a = learning rate(학습률) → 경사 하강법을 반복할 때 한 단계의 크기(=한 단계 나아갈 크기)를 결정하는 것이다.
- dJ(w) / dw = 미분계수 → 갱신할 때마다 매개변수 w에 줄 변화를 나타냄. 미분을 통해 구하며, 코드에서 dw라고 주로 표시.
- dw > 0 = 해당 점에서의 접선의 기울기가 양수 : w는 점점 작아지며 초기화한 점이 한 단계씩 내려올 것
- dw < 0 = 해당 점에서의 접선의 기울기가 음수 : w는 점점 커지며 초기화한 점이 한 단계씩 내려올 것

---

### 계산 그래프(Computation Graph)

: 특정한 출력값 변수를 최적화하고 싶을 때 사용한다.

→ 로지스틱 회귀를 사용하는 상황인 경우에는 계산 그래프가 비용 함수 J가 되는 것이다.

[신경망의 계산은 아래와 같이 나뉨]

1. 정방향 패스, 정방향 전파 : 신경망의 출력값을 계산
- 계산 그래프에서 정방향(왼쪽 → 오른쪽)으로 이동하며 출력값을 구한다.
1. 역방향 패스, 역전파 : 경사나 도함수를 계산
- 계산 그래프에서 역방향(오른쪽 → 왼쪽)으로 이동하며 미분해 도함수를 구한다.

 → **미분의 연쇄법칙**

: F(g(x))의 미분 = F'(g(x)) * g'(x) 이다.

: 입력변수 a에서 시작해 v를 거쳐 최종 출력값 J로 가는 과정이 존재한다고 가정 → 여기서 dJ/da = dJ/dv * dv/da 이다.

---

### 로지스틱 회귀의 경사 하강법을 위해 계산 그래프 활용하여 필요한 핵심 공식 구현하기

$$da = -y / a +(1-y)/(1-a)$$

$$dz = a - y$$

$$db = dL/db = dz$$

→ 하나의 훈련 샘플 구현한 것이고, 이제 m개의 훈련 샘플 존재하는 훈련 세트 사용해 로지스틱 회귀의 경사 하강법 구현해보자

**[비용 함수 활용하여 m개의 훈련 세트 사용해 경사 하강법 구현하기]**

$$(d/dw1)J(w,b) = (1/m) i=1, i = m Σ ((d/dwi) L(ai, yi)) $$

![IMG_0386.jpg](GDSC%20machine%20learning%20study%20e20bbf21ed934a95bdecd5f9eb2e8890/IMG_0386.jpg)

→ 로지스틱 회귀의 경사 하강법을 구현한 코드로, 구현에 있어 for 문이 활용되었다. 하지만, for 문은 최대한 사용하지 않는 것이 좋다. 시간 복잡도 측면에서 For문은 불리하기 때문이다. 딥러닝은 주로 아주 큰 데이터를 다룰 때가 많기에 수많은 데이터를 빨리 처리해 결과를 내는 것이 중요하다. 그렇다면 For 반복문을 활용하지 않고 어떻게 코드를 구현할 수 있을까

⇒ 벡터화를 활용하면 된다.

---

# 3. 파이썬과 벡터화

### 벡터화(Vectorization)

: 벡터화되지 않은 구현에 비해 벡터화된 구현은 W^Tx를 직접 계산한다

→ 파이썬 Numpy에서 dot 명령어를 활용하는 것이다. Ex) np.dot(w, x) = w^t, Z = np.dot(w,x) + b

### SIMD(Single Instruction Multiple Data)

: 병렬 명령어 ← 하나의 명령어로 여러 개의 값을 동시에 계산하는 방식이다. 벡터화 연산을 가능케 한다.

⇒ 많은 대규모 딥러닝 구현은 GPU에서 이루어진다. 물론, CPU에서도 가능하다. 하지만, 벡터화 연산을 가능하게 해주는 병렬 명령어인 SIMD를 통한 계산를 GPU가 CPU보다 더 잘한다. (CPU도 나쁘지는 않다.)

```python
import numpy as np
import time

a = np.random.rand(1000000)
b = np.random.rand(1000000)

tic = time.time()
c = np.dot(a, b)
toc = time.time()

print(c)
print("Vectorized version:" + str(1000*(toc-tic)) + "ms")

c = 0
tic = time.time()
for i in range(1000000):
	c += a[i]*b[i]
toc = time.time()

print(c)
print("for loop:" + str(1000*(toc-tic)) + "ms")
```

다음과 같은 행렬이 존재할 때, 이를 벡터화되지 않은 버전과 벡터화된 버전으로 구분하여 지수화 계산(exponential operation)을 하는 코드를 작성해라.                                                                                           →                                                        v = [v_1 ... v_n]

```python
import numpy as np
import math

## v = np.array([v1, v2, ..., vn])

# non vectorized version -> for loop version
u = np.zeros((n,1))    # 크기 n*1(세로*가로) 행렬 u를 0으로 초기화한다
for i in range(n):
	u[i] = math.exp(v[i])

# vectorized version
u = np.exp(v)    # 행렬 v의 모든 원소를 지수화 할 때 -> np.exp(행렬이름)
```

⇒ 위에서는 지수화 예시를 본 것이고 실제로 NumPy 라이브러리에는 다른 벡터 함수가 많다.

1. 지수화 → np.exp(행렬이름)
2. 로그값 → np.log(행렬이름)
3. 절대값 → np.abs(행렬이름)
4. 값 비교(큰 것) → np.max(행렬이름, 0) : 행렬의 원소와 0 중 더 큰 값 반환한다.(0 말고 다른 것도 O)
5. 제곱값 → 행렬이름**2 : 행렬의 모든 원소를 제곱한 벡터를 반환한다.
6. 역수값 → 1/행렬이름 : 행렬의 모든 원소의 역수로 이루어진 벡터를 반환한다.

→ for loop를 통해 어떤 벡터 계산을 할 때는, 먼저 그러한 계산이 가능한 NumPy 내장함수가 존재하는지 살펴보고 그것을 활용한다. 

---

**로지스틱 회귀와 경사 하강법에 적용하기**

![IMG_0388.jpg](GDSC%20machine%20learning%20study%20e20bbf21ed934a95bdecd5f9eb2e8890/IMG_0388.jpg)

: 원래 for문 2번 활용됨(m개의 훈련 샘플 지닌 훈련 세트 및 1~nx)

→ dw = np.zeros(nx, 1)와 dw += x^(i)*dz^(i)를 적고, dw/=m을 대신 적음으로서 벡터화하는 것을 통해 두 번째 for문 삭제할 수 있다.

---

**[로지스틱 회귀를 벡터화하여 전체 훈련 세트에 대한 경사 하강법 구현 시 for 문 아예 사용하지 않기]**

1. **정방향 전파**

Z = np.dot(w.T, x) + b

- b는 (1, 1) 행렬이라고 할 수 있는 실수이다,
- 파이썬에서 벡터와 실수를 더할 때: 벡터의 크기가 m일 때, 실수를 (1, m) 크기의 행렬(벡터)로 바꾼다.
    
    → 이 연산을 파이썬에서 "**브로드캐스팅(Broadcasting)**"이라고 한다.
    
- 대문자 Z는 z^(1)부터 z^(m)까지를 포함하는 (1, m) 행렬이 된다.

A = S(Z)

- a는 예측값을 나타낸다.
- 파이썬에서 벡터 값을 가지는 시그모이드 함수를 구현한다.
- 구현한 시그모이드 함수에서 대문자 Z를 입력으로 받아 출력으로 대문자 A를 반환한다.
- 대문자 A는 a^(1)부터 a^(m)까지를 포함하는 (1, m) 행렬이 된다.
1. **역방향 전파**

     **[m개의 훈련 샘플 지닌 훈련 세트에 대해 벡터화하여 경사 계산 구현하기]**

db = (1/m) * np.sum(dZ)

- np.sum(dZ)는 dz^(1)부터 dz^(m)까지를 포함하는 (1, m) 행렬을 모두 더한 값을 반환한다.
- db는 반환된 값을 훈련 샘플 총 개수 m으로 나눈다.

dw = (1/m) * X * dZ^T

- dw는 1/m에 (n_x, m) 벡터와 (m, 1) 벡터를 곱한 것과 같다. 따라서 위와 같다.
- 경사 하강법 갱신은 w := w - ∂*dw, b := b - ∂*db

⇒ 위 두 가지 식을 통해 이제 for 문 없이 변수의 갱신값을 계산할 수 있다.

**[벡터화된 로지스틱 회귀 구현하기]**

- 경사 하강법 구현에서 두 개의 for 문 모두 제거하는 방법 배움. 단, 이것은 한 번의 경사 계산에 대해서임
- 경사 하강법을 n번 반복할 때, n번 반복하는 for 문 필요함

---

### 브로드캐스팅

- 파이썬 코드 실행 시간을 줄일 수 있는 또 하나의 방법

cal = A.sum(axis=0)

- sum(axis=0) → 세로축, 즉, 열을 더하는 것
- sum(axis=1) → 가로축, 즉, 행을 더하는 것

percentage = 100*A/(cal.reshape(1, 4))

- A는 (3, 4) 행렬일 때, 윗줄은 (3, 4) 행렬인 A를 (1, 4) 행렬로 나누는 것이다.
- 행렬의 차원이 정확하지 않을 때 reshape 사용해 원하는 차원으로 맞추기

⇒ 어떻게 (3, 4) 크기의 행렬을 (1,4) 행렬로 나눌 수 있을까?

: (1,4) 행렬을 3번 복사해 (3,4)로 만들어주고 계산한다.

(m,n) 행렬과 (1, n) 행렬에 대해 사칙연산을 한다면   (1, n) → (m, n)                                                       (m,n) 행렬과 (m, 1) 행렬에 대해 사칙연산을 한다면 (m,1) → (m, n)                                                      (m,1) 행렬과 실수에 대해 사칙연산을 한다면 실수 → (m,1)                                                                    (1,m) 행렬과 실수에 대해 사칙연산을 한다면 실수 → (1, m)

---

### Python NumPy

```python
import numpy as np

a = np.random.randn(5)    # 가우시안 분포 따르는 변숫값 5개를 배열에 저장한다.

print(a)
print(a.shape)    # (5,) 크기가 5이다. 랭크가 1인 배열(= 형 or 열 벡터 모두 아님)이라고 부른다.
print(a.T)        # 전치 출력해도 a 그대로 출력됨
print(np.dot(a, a.T))   # a 전치와 a를 곱하니 행렬이 나와야 된다고 생각하지만 실제로 값이 나옴

a = np.random.randn(5,1)    # 5*1형 배열이 됨
print(a)
print(a)           # 지금 a의 전치는 행 벡터가 됨(1*5 행렬)
print(np.dot(a, a.T))   # 행렬이 결과값으로 나옴

```

> a = np.random.randn(5)
> 
- a.shape = (5,) : 행 벡터도 열 벡터도 아닌 랭크가 1인 배열 ⇒ 결과가 직관적이지 않게 된다.
- 신경망 구현할 때 랭크 1 배열 아예 사용하지 않는 것이 좋다. (프로그래밍 과제 할 때도 마찬가지)
- a = np.random.randn(5, 1) → a.shape = (5,1) ⇒ column vector(열 백터) or (5,1) 행렬
- a = np.random.randn(1, 5) → a.shape = (1,5) ⇒ row vecotr(행 벡터) or (1, 5) 행렬

> 코드에서 벡터의 차원을 확실히 알지 못 할 때
> 

assert (a.shape == (5, 1))

→ a가 (5,1) 열 벡터라는 것을 확실히 하기 위해 assert 함수 사용한다.

- 랭크 1 배열을 얻으면 reshape 함수 사용해서 원하는 열 벡터, 행 벡터 형태로, 즉, 필요한 차원 형태를 가진 행렬, 벡터로 바꿔준다.

---

### 로지스틱 회귀에서 비용 함수 사용하는 이유

p(y|x) = (^y)^y * (1-^y)^(1-y)

- y: 예측값, x: 입력 특성

→ 우리는 예측값의 정확도를 올려야 한다. 즉, p(y|x)의 값 높아야 한다.

⇒ log 함수를 활용해보자!

log p(y|x) = log (^y)^y * (1-^y)^(1-y) = y * log (^y) + (1-y) * log (1-^y) = - L(^y, y)

- log 함수는 강한 단조 증가 함수 → log p(y|x)를 최대화한다 = p(y|x)를 최대화한다
- log p(y|x)는 손실 함수의 음수가 된다 ⇒ 로지스틱 회귀에서는 손실을 "최소화"하고 싶기 때문

---

# 4. 얕은 신경망 네트워크

> 신경망의 층 세기
> 

: 입력층을 제외한 은닉칭+출력층의 수로 센다. ⇒ 은닉층 첫 번째가 천 번째 층, ..., 출력층이 마지막 층

### 2층 신경망

![IMG_0389.jpg](GDSC%20machine%20learning%20study%20e20bbf21ed934a95bdecd5f9eb2e8890/IMG_0389.jpg)

- 신경망의 입력 층 : 입력 특성 x1, x2, x3 존재하는 층
- 신경망의 은닉 층
- 신경망의 출력층: 노드 하나로 이루어지고 예측값인 ^y를 계산한다.

⇒ 훈련 세트에서 입력 및 출력값은 알 수 있지만 은닉층의 값은 알 수 없다. 따라서 '은닉'층인 것이다.

- 입력값의 또 다른 표기 → a^[0] ⇒ a : 활성값을 의미, 신경망의 층들이 다음 층으로 전달해주는 값을 의미
    
    ⇒ 입력층의 활성값이라고 부름
    
- a^[1] ⇒ 은닉층의 활성값 :  위 그림 상황에서는 노드가 4개 존재 → a^[1]은 (1,4) 행렬, 열 벡터이다.
    
    → a^[1]_1, a^[1]_2, ...
    
- 출력층 → ^y = a^[2]

---

### 신경망에서 정확히 어떻게 출력값을 계산하는가

: 로지스틱 회귀와 비슷하지만, 여러 번 반복된다.

→ 이것을 벡터화하는 것도 중요한데, 어떻게 할까?

1. 벡터였던 소문자 x 열로 쌓아 행렬 X 얻는다. → X : 훈련 샘플이 열로 쌓은 행렬 → (nx, m) 행렬이 된다.
2. z에도 같은 작업 → 열 벡터인 Z^[1] : z^[1](1), z^[1](2), ..., z^[1](m)
3. a도 마찬가지로 A^[1] : a^[1](1), a^[1](2), ..., a^[1](m)

→ 행렬 왼쪽 위에 있는 값은 첫 은닉 유닛의 첫 훈련 샘플의 활성값이 되고, 바로 아래 값은 첫 훈련 샘플의 두 번째 은닉 유닛의 활성값이 될 것이다.

[행렬 Z, A]

- 행렬의 세로 = 은닉 유닛의 번호
- 행렬의 가로 = 은닉 유닛은 고정, 훈련 샘플이 바뀜

[행렬 X]

- 행렬의 세로 = 서로 다른 입력 특성. 즉, 신경망 입력층의 다른 노드들!
- 행렬의 가로 = 서로 다른 훈련 샘플

→ 기억해야 할 것: X = A^[0] : 입력 특성은 신경망의 입력층에서 입력된 것이기에 A^[0]과 같다.

---

> 신경망 만들 때 어떤 활성화 함수를 사용할까? : 은닉층과 출력층에서 사용할 함수 정하기
> 

: 지금까지는 시그모이드 함수 사용함 → 하지만, 다른 함수가 더 좋은 선택일 경우도 존재함

- 다른 층에 다른 활성화 함수 사용할 수 있다.
1. sigmoid
- 0부터 1
- 평균값 0.5
- 시그모이드 사용하는 예외적 상황 ⇒ 이진 분류, 출력층(y가 0 아니면 1이므로 값이 0~1인 것이 -1~1인 것보다 유리하다.)
1. tanh(쌍곡 탄젠트 함수)
- 값은 -1 ~ +1
- 공식  a = (e^x - e^(-x)) / (e^x + e^(-x))
- 수학적으로 시그모이드 함수 조금 옮긴 함수이다 → 비슷하지만, 원점 지나고 비율이 달라짐
- 은닉 유닛에 대해 탄젠트로 놓으면 거의 항상 시그모이드 함수보다 좋음(값이 -1과 1 사이라 평균값이 0에 더 가깝기 때문이다.)

→ 시그모이드와 탄젠트 함수 단점: 함수 입력값 x가 매우 크거나 작으면 함수의 도함수가 굉장히 작아진다. 즉, 함수의 기울기가 0에 가까워지고, 이는 경사 하강법 수행 속도를 느려지게 한다.

1. ReLU
- 머신러닝에서 인기 있는 함수
- 공식 a = max(0.01x, x)
- x > 0 : 도함수 1
- x = 0일 때는 엄밀하게 도함수 정의되지 않았지만, 그럴 확률 극히 적으며 그냥 도함수 1이나 0으로 가정해도 잘 작동한다.
- x < 0 : 도함수 0

→ ReLU 함수의 단점: x < 0 일 때, 도함수가 0이라는 것 → 이것을 고려한 다른 버전인 leaky ReLU가 존재

1. leaky ReLU
- x < 0 일 때, 도함수가 0인 대신 약간의 기울기를 준다.
- 실제로 많이 쓰이지 않는다. 하지만, ReLU보다 좋은 결과 도출한다.

→ ReLU, leaky ReLU 함수 장점 : 대부분의 x에 대해 기울기가 0이 아니다. ⇒ 빠른 신경망 학습이 가능함

이진 분류의 출력층에는 시그모이드 함수, 다른 경우에는 ReLU가 활성화 함수의 기본값으로 많이 사용된다. 은닉층에 어떤 함수를 써야 할지 모르겠다면 ReLu 사용하기. 때때로 tanh도 사용된다.

⇒ 기억해야 할 것

1. 학습을 느리게 하는 것 = 기울기 0
2. 가장 많이 쓰는 활성화 함수는 ReLU
3. 함수 고르지 못하겠으면 ReLU, leaky ReLU 사용하기
4. 다양한 환경의 신경망 구현하고 결과 비교해서 최상의 것 선택해도 괜찮다.

⇒ 궁긍즘: 왜 굳이 활성화 함수 사용할까?

---

### 신경망이 비선형 활성화 함수를 필요로 하는 이유

[선형 활성화 함수, 항등 활성화 함수]

- 신경망은 입력의 선형식 만을 출력함
- 층이 많은 심층 신경망의 경우 층이 얼마나 많던 신경망은 선형 활성화 함수만 계산하기에 은닉층이 없는 것과 다름 없다.
- 선형 활성화 함수 사용하고 여기에 시그모이드 함수 사용하면 표준 로지스틱 회귀에서 더 나아질 수 없음
- 선형 은닉층은 쓸모가 없다 ⇒ 두 개의 선형 함수 조합 = 한 개의 선형 함수
- 선형 활수화 함수인 g(z) = z → 회귀 문제에 대한 머신러닝 시 사용한다.(y가 실수일 경우)
- 선형 활성화 함수를 은닉층에 사용하는 경우는 드물고, 대부분 출력층에만 사용한다.

⇒ 즉, 비선형식이 있어야지 신경망 층이 많아졌을 때 더 효율적이고 좋은 계산이 가능하다.

---

### 활성화 함수의 미분

: 역전파를 위해서라면 함수의 도함수를 구할 수 있어야 한다.

- 입력 변수 x에 대한 함수 ƒ의 도함수 =  ƒ'(x)
1. 시그모이드 활성화 함수

S(x) = 1 / (1 + e^(-x))

- 어떤 점 x에 대해 S(x)의 기울기 = (d / dx) * S(x) = S(x) * (1 - S(x)) = slope of S(x) at x
- 신경망에서 : a = S(x) → S'(x) = a * (1 - a)
1. tanh 활성화 함수

T(x) = tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))

- 어떤 점 x에 대해 T(x)의 기울기 = (d/dx) * T(x) = 1 - {T(x)}^2
- 신경망에서 : a = T(x) → T'(x) = 1 - a^2
1. ReLU 활성화 함수

R(x) = max(0, x)

- 도함수 R'(x)는  x < 0일 때는 0, x > 0일 때는 1이 된다.
- x = 0일 때 도함수는 1, 0 중 아무거나 택해도 잘 됨 → 정확히 x = 0 될 확률 진짜 작기 때문이다.
1. Leaky ReLU 활성화 함수

LR(x) = max(0.01x, x)

- 도함수 LR'(x)는 x < 0일 때 0.01, x > 0일 때는 1이 된다.
- x = 0일 때 도함수는 0.01, 1 중 아무거나 택해도 잘 됨 → ReLU와 동일한 이유

---

### 경사 하강법 구현하기

**단일층 신경망**

: 4개의 식 필요 → dw^[1], db^[1], W^[1] (:=으로 변수 갱신), b^[1]   (:=으로 변수 갱신)

**2개 이상의 층으로 구성된 신경망**

: 단순히 1이 아니라 1~m까 반복하기

1. **정방향 전파**
- Z^[1] = w^[1] * X + b^[1]     (X = A^[0])
- A^[1] = S(Z^[1])
- Z^[2] = w^[2] * A^[1] + b^[2]
- A^[2] = S(Z^[2])
1. **역방향 전파**
- dz^[2] = A^[2] - Y
- dw^[2] = (1/m) * dz^[2] * A^[1]^T
- db^[2] = (1/m) * np.sum(dz^[2], axis = 1, keepdims = True)
- dz^[1] = W^[2]^T * dz^[2] * g^[1]'(Z^[1])
- dw^[1] (1/m) * dz^[1] * X^T
- db^[1] = (1/m) * np.sum(dz^[1], axis = 1, keepdims = True)

→ db^[1]은 (n^[1], 1) 행렬

→ sum은 axis = 0 or axis = 1 사용해 각각 세로, 가로로 더하는 연산 수행

→ keepdims는 전에 봤던 (5,)와 같은 랭크 1 배열 되지 않도록 해주는 역할 수행

→ keepdims 대신 reshape 사용해도 된다.

---

### 신경망에서의 초기화

- W를 0으로 초기화하지 않는다 → 은닉 유닛이 여러 개 있어도 한 개 있는 것과 같은 상황 벌어짐
    
    → 대칭이 되어 같게 되는 것임
    
    ⇒ W^[1] = np.random.randn((2,2)) * 0.01
    
    → 왜 0.01을 곱할까?
    
    : 초기값을 정할 대는 매우 작은 값으로 하는 것이 좋다. tanh, sigmoid 함수를 사용할 때 가중치가 너무 크면 양 끝으로 가서 함수의 기울기가 0에 가까운 경우가 많아지고, 학습의 속도가 느려지기 때문이다.
    
    : 은닉 유칭 층이 적은 얕은 신경망에서는 0.01 곱해도 되지만, 매우 깊은 깊이의 신경망은 0.01과는 다른 수를 선택해도 괜찮다. ⇒ 보통은 작은 수가 되는 쪽으로!
    
- b는 0으로 초기화해도 된다 → 대칭의 문제 가지지 않기 때문이다.
    
    ⇒ b^[1] = np.zero(2,1))
    

---

# 5. 심층 신경망 네트워크

### 더 많은 층의 심층 신경망

**설명**

- 우리가 살펴본 로지스틱 회귀는 매우 얕은 모델이다. 은닉층이 많으면 많을수록 깊은 신경망이라고 한다.("deep")
- N layer 신경망 → 은닉층: N-1, 출력층: 1 (개)

**[표기법]**

1. L = 층의 개수 (은닉층+출력층)
2. n^[l] = # units in layer l
3. a^[l] = # actications in layer l = g^[l](z^[l])     (g: 활성화 함수)
4. n^[0] = n_x = 입력 특성 총 개수
5. 입력 특성 x = a^[0]
6. 예측값 ^y = a^[L]

---

**정방향 전파**

```python
# Z = w^TX + b, x = A^[0], b는 실수 
for l in range(1, n):              # 여기서는 명시적인 for문 필요함
	Z[l] += W[l]*A[l-1] + b[l]
	A[l] = g[l](Z[l])

^Y = A[n]
```

---

### 행렬의 차원 알맞게 하기

Z[1] = W[1]*X + b[1]

→ Z[1] : (n[1], m) 행렬/벡터, W[1] : (n[1], n[0]) 행렬/벡터, X : (n[0], m) 행렬/벡터, b[1] : (n[1], m) 행렬/벡터

[정리]

z[l], a[l] : (n[l], 1)                                                                                                                              Z[l], A[l] : (n[l], m)           l=0 → A[0] = X : (n[0], m)                                                                       dZ[l], dA[l] : (n[l], m)

---

### 심층 신경망-얕고 깊음에 따라

1. 네트워크가 깊어지면 깊어질수록, 더 많은 특징을 찾아낼 수 있다.
- 첫 은닉층: 상대적으로 간단한 계산하고 간단한 특징 찾아냄 → 마지막 은닉층: 탐지된 간단한 것들을 모아 복잡한 특징 찾아낸다.

      Ex) input = 사람 얼굴 사진 → 모서리, 코/눈..., 얼굴

- 노드의 개수 = 네트워크의 게이트 수
1. 회로 이론: 상대적으로 은닉층의 개수가 작지만, 깊은 심층 신경망에서 사용할 수 있는 함수가 존재한다. 
    
    → 얕은 네트워크로 같은 함수를 활용하려고 하면, 계산하기에 충분한 은닉층이 없기 때문에 기하급수적으로 많은 은닉 유닛이 필요하다.
    

---

### 심층 신경망-네트워크

[정방향 전파와 역방향 전파 그림으로 이해하기]

![IMG_0394.jpg](GDSC%20machine%20learning%20study%20e20bbf21ed934a95bdecd5f9eb2e8890/IMG_0394.jpg)

- cache에는 Z의 값뿐만 아니라, W, b도 저장되어 있다
- cahce는 필요할 곳에 복사하여 사용한다.
- 정방향 전파에서는 X를 초기화
- 역방향 전파에서는 dA[l]을 초기화해야 함. dA[l] = - (y^(1)/a(i)) + (1-y^(1)) / (1-a^(1)) + ... - (y^(m)/a^(m)) + (1-y^(m) / 1 - a^(m))

: X input 받고, 층 계속해서 넘어가면서 a^(i) 구하고, 정방향 전파 과정에서 최종적으로 a^(l)을 구한다. 그리고 역방향 전파는 l번째 층에서 시작되어 da[l]이 첫 입력이고, dW[l], db[l], da[l-1]을 출력한다. 이때, 정방향 함수 계산에서 저장해둔 캐시를 사용한다.

---

### 하이퍼파라미터

cf) 변수란 신경망에서 학습할 수 있는 W와 b를 의미한다.

하이퍼파라미터

1. 학습률 α(learning rate α)
2. 반복횟수(numbers of iteration)
3. 은닉층의 개수  → L
4. 은닉유닛의 개수 → n^[1], n^[2], n^[3], ...
5. 활성화 함수의 선택 → sigmoid, tanh, ReLU, leaky ReLU

→ 1~5의 하이퍼파라미터는 궁긍적으로 매개변수 W와 b를 통제한다. = 최종 매개변수 W, b의 최종값을 결정함

1. 모멘텀항
2. 미니배치 크기
3. 다양한 형태의 정규화 매개변수
- 딥러닝은 아주 많은 하이퍼파라미터 지님. 초반에는 그냥 매개변수라고 불렀는데, 사실은 "진짜 매개변수(W,b)"를 정하는 매개변수라고 봐야 하는 것임
- 하이퍼파라미터는 정해지지 않았다. 우리가 원하는 값 넣어가면서 결과를 보고 조정하면서 최종 값을 결정한다.

---

### 인간의 뇌 - 신경망

: 이 비교는 시간이 지날수록 무너지고 있음. 즉, 관계가 그렇게 크지 않다는 것이다. 뉴런 하나를 보았을 때, 뉴런 하나가 하는 일을 신경과학자들조차 모름 → 매우 복잡. 이러한 뉴런 하나를 단일 로지스틱 회귀와 비교하라 때, 뉴런의 '일부'가 단일 로지스틱 회귀를 사람들에게 설명하는 데 있어 도움이 되는 것이다.

⇒ 미래로 갈수록 인간의 뇌와 신경망 사이 비교는 무의미해질 것이다.