# 2. Regularizing your neural network

### **Recularization**

- 정규화: 높은 분산으로 신경망이 데이터를 과대적합하는 문제 의심될 경우 첫번재로 사용
- **Logistic Regression**
    
    ![Untitled](2%20Regularizing%20your%20neural%20network%20903aee334c11432a9c20aa94d8ee4fda/Untitled.png)
    
    - $\lambda$: 정규화 매개변수(regularization parameter)
        - 설정이 필요한 또 다른 하이퍼파라미터 ⇒ 파이썬의 명령어이기도 함
        - 개발 세트 혹은 교차 검증 세트를 주로 사용
        - 다양한 값을 시도해서 훈련 세트에 잘 맞으면서 두 매개변수의 norm을 잘 설정해 과대적합을 막을 수 있는 최적의 값을 찾음
    - 매개변수 $w$만 정규화하는 이유 : 보통 $w$는 높은 차원의 매개변수 벡터이므로 높은 분산을 가질때 많은 매개변수를 가짐.
        
        but, b는 하나의 매개변수이기 때문에 생략함
        
    - L1 정규화
        - **목표: 모델 압축**
        - 사용시 $w$는 희소해짐 = $w$벡터 안에 0이 많아진다는 의미
        - 모델을 압축하는데 도움이 됨 ⇒ 매개변수가 0일 경우 메모리가 적게 필요하기 때문
        - but 모델을 희소하게 만들기 위해 L1 정규화를 사용하는 것은 큰 도움이 되지 않음
        - **네트워크 훈련시 L2정규화 사용**
- **Neural Network**
    
    ![Untitled](2%20Regularizing%20your%20neural%20network%20903aee334c11432a9c20aa94d8ee4fda/Untitled%201.png)
    
    - Frobenius norm : 행렬의 원소 제곱의 합
        
        ![Untitled](2%20Regularizing%20your%20neural%20network%20903aee334c11432a9c20aa94d8ee4fda/Untitled%202.png)
        
        - 행렬의 L2 norm
    - L2 정규화 = weight decay라고 불림
        
        ![Untitled](2%20Regularizing%20your%20neural%20network%20903aee334c11432a9c20aa94d8ee4fda/Untitled%203.png)
        
        - weight에 1보다 작은 값이 곱해지기 때문

### **Why Regulariztion Reduces overfitting**

- $\lambda$값을 크게 만들어서 가중치행렬 $w$를 0에 가깝게 설정할 수 있음.
    
    ![Untitled](2%20Regularizing%20your%20neural%20network%20903aee334c11432a9c20aa94d8ee4fda/Untitled%204.png)
    
    - 많은 은닉 유닛을 0에 가까운 값으로 설정해서 은닉 유닛의 영향력을 줄임
        
        ⇒ 간단하고 작은 신경망이 되어 과대적합이 덜 일어남
        
        ⇒ Logistic Regression과 유사하게 만들게됨
        
- $tanh$ 활성화 함수를 사용했을 경우 $\lambda$ 값이 커짐 → 비용함수에 의해 $w$는 작아짐
    
    ⇒ 이 때, 아래 식에 의해 z도 작아지게 됨.
    
    ![Untitled](2%20Regularizing%20your%20neural%20network%20903aee334c11432a9c20aa94d8ee4fda/Untitled%205.png)
    
    ![Untitled](2%20Regularizing%20your%20neural%20network%20903aee334c11432a9c20aa94d8ee4fda/Untitled%206.png)
    
    - $g(z) = tanh(z)$
    - z가 작을 때 g(z)는 선형 함수가 되고, 전체 네트워크도 선형이 되기에 과대적합과 같이 복잡한 결정을 내릴 수 없음.
    - 정규화 매개변수가 매우 크면 매개변수 $w$는 매우 작음
    - $b$의 효과를 무시하면 z는 상대적으로 작고 작은 범위의 값을 가짐
- tip
    - 경사하강법을 구현할 때 경사 하강법의 반복의 수에 대한 함수로 비용함수를 설정
    - 비용함수 J가 경사 하강법의 반복마다 단조감소하기를 원할 것
    - 경사 하강법을 디버깅 할 때는 두번째 항을 포함한(정규화를 적용한) 새로운 비용함수 J를 쓰는것이 좋음

### **Dropout regularization**

- Dropout
    - L2 Regularization 외의 또 다른 정규화기법
    - 신경망의 각각의 층에 대해 노드를 삭제하는 확률을 설정하는 것
        - 노드를 삭제할 경우 삭제된 노드의 들어가는 링크와 나가는 링크 모두 삭제
        - 더 작고 간소화된 네트워크가 됨 ⇒ 하나의 샘플을 역전파로 훈련시킴, 다른 샘플에서도 다른 세트의 노드들을 삭제
        - 각각의 샘플에서 더 작은 네트워크를 훈련시키는 방식
    - 
- Dropout 구현 방법
    - Inverted Dropout(역 드롭아웃)
        
        ![Untitled](2%20Regularizing%20your%20neural%20network%20903aee334c11432a9c20aa94d8ee4fda/Untitled%207.png)
        
        - keep_prob : 어떤 은닉 유닛이 유지될 확률
        - keep_prob을 나누는 이유: a3의 기댓값을 유지하기 위해
        - 신경망을 평가할 때, Inverted Dropout 기법이 테스트를 쉽게 만들어줌
        - 반복을 통해 경사 하강법의 하나의 반복마다 0이 되는 은닉 유닛들이 달라짐
- Making Predictions at test time
    
    ![Untitled](2%20Regularizing%20your%20neural%20network%20903aee334c11432a9c20aa94d8ee4fda/Untitled%208.png)
    
    - X : test의 샘플
    - 테스트에서는 '예측'을 하는 것이므로, 결과가 무작위로 나오는 것을 원하지 않음
    - 테스트에 드롭아웃을 구현하는것 ⇒ 노이즈만 증가시킬뿐 비효율적임

### **Understanding Dropout**

- Dropout : 노드를 삭제함으로써 더 작은 신경망에서 훈련시키는 것이므로 정규화의 효과를 주는 것처럼 보임
- 단일 유닛의 관점
    - 유닛이 해야 하는 일: 입력을 받아 의미있는 출력을 생성하는 것
    - 드롭아웃을 통해 입력은 무작위로 삭제될 수 있음
    - 특정 입력에 의존할 수 없으므로, 가중치를 다른 곳으로 분산 시키는 효과가 있음
- 층마다 keep_prob의 값을 다르게 설정할 수 있음.
    - 매개변수가 적은 층 : keep_prob의 값을 높여도 됨
        - cf. keep_prob = 1.0 ⇒ 모든 유닛을 유지하고 해당 층에서는 드롭아웃을 사용하지 않는다
    - 매개변수가 많은 층(과대적합의 우려가 많은 층)
        - 더 강력한 형태의 드롭아웃을 위해 keep_prob 값을 작게 설정
    - 입력층에도 드롭아웃 적용할 수 있으나, 지양하는 것이 좋음(입력특성을 최대한 살리기 위해)
    - tip : Computer vision에서는 데이터의 양이 충분하지 않으므로. 과대적합일 경우에만 사용
- 단점
    - 비용함수 J가 잘 정의되지 않음.
    - 모든 반복마다 무작위로 한 뭉치의 노드들을 삭제
        
        $\therefore$ 경사 하강법의 성능을 이중으로 확인한다면 모든 반복에서 잘 저으이된 비용함수 J가 하강하는지 확인하기 어려워짐
        
        ⇒ 최저고하하는 비용함수가 잘 정의되지 않아 계산하기 어렵기때문
        
    - 우선 드롭아웃을 사용하지 않고, 비용함수가 단조감소인지 확인 후 사용해야함.
    - 드롭아웃을 사용할때 코드를 바꾸지 않도록 해야함.

### **Other Regularization Methods**

- **Data augmentation**
    - 보통 이미지를 대칭, 확대, 왜곡 혹은 회전을 시켜서 새로운 훈련 데이터를 만든다.
    - 완전히 새로운 샘플을 얻는것보다 더 많은 정보를 추가해주는 것은 아님,
- **Early Stopping**
    - 훈련 오차나 비용함수 J는 단조 감소하는 형태로 그려져야 함.
    - Early Stopping : 개발 세트 오차도 함께 그려줌
        
        ![Untitled](2%20Regularizing%20your%20neural%20network%20903aee334c11432a9c20aa94d8ee4fda/Untitled%209.png)
        
        - dev set error가  중간에 하락하지 않고 증가하기 시작
            - 과대적화가 시작되는 시점
            - 이 때 Early Stopping이 개발 세트의 오차 저점 부근에서 훈련을 멈춤
        - iteration이 커질수록 $w$역시 증가하는데, 중간에서 멈춘다면 $w$역시 중간값을 가짐
            
            ⇒ L2 정규화와 비슷하게 매개변수 $w$에 대해 더 작은 norm을 갖는신경망을 선택함으로써 Overfitting을 줄임
            
    - 단점 : 비용함수 J 최적화 작업과 과대적합을 줄이는 작업을 독립적으로 작업할 수 없음
        - 머신러닝 과정 → 서로 다른 몇 가지 단계로 이루어짐 ⇒
            1. 비용함수 J를 최적화하는 알고리즘(경사하강법, 모멘텀, RMSProp, Adam～)
            2. 최적화 후 과대적합을 막는 몇 가지 도구(정규화, 데이터 추가)
            3. 많은 하이퍼파라미터 > 선택과정은 매우 복잡함
            
            $\therefore$ 비용함수 J를 최적화하는 하나의 도구 세트만 있다면 머신러닝이 더 간단해짐
            
            ⇒ 집중: $w$와 $b$를 찾는 것 ⇒ $J(w,b)$가 가능한 작아지는 값을 찾는 것만 신경씀
            
        - 과대적합을 막는 것(분산을 줄이는 것) ⇒ 별개의 도구들이 필요
        - 경사하강법을 일찍 멈춤으로써 비용함수 J를 최적화하는 것을 멈추게 됨
        - Early Stopping의 대안: L2 정규화 사용
            - 가능한 오래 신경망을 훈련시킬 수 있음.
            - 하이퍼파라미터의 탐색 공간이 더 분해하기 쉽고 찾기 쉬워짐
            - BUT 매개변수 $\lambda$에 많은 값을 시도해야함 ⇒ 컴퓨터적으로 비용이 많이 듦.
    - 장점
        - 경사 하강법 과정을 한 번만 실행해서 작은  $w$, 중간 $w$, 큰 $w$의 값을 얻게 됨
        - 많은 시도를 하지 않아도 한 번에 얻어낼 수 있음.