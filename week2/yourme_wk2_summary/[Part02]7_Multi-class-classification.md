# 7. Multi-class classification

### **Softmax Regression**

- Logistic Regression을 일반화한 Regression
- 여러개의 Class 분류시 사용됨.
    
    ![Untitled](7%20Multi-class%20classification%20f3190b9a996c4bac8077730af26bdb9f/Untitled.png)
    
    - 예측 결과값 = (4,1)차원의 벡터가 됨.
- Softmax Layer
    
    ![Untitled](7%20Multi-class%20classification%20f3190b9a996c4bac8077730af26bdb9f/Untitled%201.png)
    
    ![Untitled](7%20Multi-class%20classification%20f3190b9a996c4bac8077730af26bdb9f/Untitled%202.png)
    
    - 마지막 층의 출력값이 주어졌을 때 해당 클래스에 속할 확률을 구할 수 있음.
        - 마지막 선형 출력값(z)를 각각 지수화 시켜 임시변수 t = e^z를 구함
        - 그 후 모든 값들의 합이 1이 될 수 있도록 t 값들의 합을 나누어 정규화시킴
    - 입력값과 출력값 모두 벡터임 ⇒ (4,1) 벡터를 받아서 (4,1) 벡터를 내놓음
        - 이전에는 활성화 함수가 하나의 실수값을 받음
    - 클래스 간의 경계가 선형을 이루고 있음

### Training a S**oftmax Classifier**

- 비용함수
    
    ![Untitled](7%20Multi-class%20classification%20f3190b9a996c4bac8077730af26bdb9f/Untitled%203.png)
    
    - 학습 알고리즘이 경사하강법을 이요하여 손실 함수의 값을 작게 만듦
        
        ⇒ y^_2의 값을 가장 크게 만들어야 함
        
    - 일반적으로 손실 함수는 훈련 세트에서 관측에 따른 클래스가 무엇이든 그 클래스에 대응하는 확률을 가능한 한 크게 만듦
- Gradient descent with softmax
    
    ![Untitled](7%20Multi-class%20classification%20f3190b9a996c4bac8077730af26bdb9f/Untitled%204.png)