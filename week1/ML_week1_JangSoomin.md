# 머신러닝 1주차
## 머신러닝 모델과 데이터
### 1. 신경망 네트워크 종류
* Standard NN: real estate , online advertising
* Convolution NN: photo tagging
* Recurrent NN: speech recongnition , machine translation
### 2. 데이터 종류
* structured data: Database
* unstructured data: Audio , Image, text
### 3. 딥러닝 성능 향상 요소
* Data 증대
* Computation 향상
* Algorithm 개발
## 신경망 네트워크 기본 원리 with Logistic regression
### 1.뉴런이란
* input [features] -> o -> output [predict]이 간단한 신경망 네트워크라 할 때 o가 뉴런이다.
* 즉 데이터의 feature 값이 수많은 뉴런 층을 통해 predict값을 내놓는 것이 신경망 네트워크 모델이다.
### 2. 가장 간단한 신경망 logistic regression
  $$Y_{predict}=sigmoid(W^{T}*X+B)$$
* 목표 : 예측 값과 실제 값이 일치 하도록 하는  weight vector 와 bias를 구한다.
* 방법 : loss fuction을 최소로 하는 값 즉 cost function을 0으로 만드는 값을 찾는다.
* 경사 하강법이란 도함수를 방향으로 a를 step size로 하여 미분 값이 0인 지점을 찾는 것이다.  
  이를 위해서 도함수 와 적절한 a 값이 필요 하다.
### 3. 데이터 벡터화
* input data : N-features , M-training sets    　 $x_{n}^{m}$  
   $$X=\begin{bmatrix}x_{1}^{1}&...&x_{1}^{M}\\&.\\&.\\&.\\x_{N}^{1}&...&x_{N}^{M}\\ \end{bmatrix}$$
* output data : M-training sets    　    $y^{m}$  
   $$Y=\begin{bmatrix}y^{1}&...&y^{M} \end{bmatrix}$$  
* weight : N-features     　 $w_{n}$  
   $$W=\begin{bmatrix}w_{1}\\.\\.\\.\\w_{N}\\ \end{bmatrix}$$ 
### 4. 활성화 함수 
* sigmoid : 주로 이진분류 모델, 출력층에 사용된다.
$$1\over 1+e^{-x}$$
* tanh : 대부분의 경우 sigmoid보다 성능이 좋으며 은닉층에서 사용한다.  
   $$e^{x}-e^{-x}\over e^{x}+e^{-x}$$
* Relu : sigmoid와 tanh는 층이 깊을 수 록 도함수가 0에 가까워진다는 문제점을 보완하는 함수이다.  
  거의 모든 모델의 기본 활성화 함수로 샤용된다.  
  $$max(0,x)$$
* 선형 함수는 증폭 효과가 없어 활성화 함수로 비선형 함수를 사용한다.
## 신경망 네트워크 학습
### 5.신경망 네트워크 벡터 notation
* data: N-features M-training set  
  layer: L-layers
* $X^{m}=\begin{bmatrix}x^{1}\\.\\.\\.\\x^{n}\\ \end{bmatrix}$
  $a^{l-1}=\begin{bmatrix}a_{1}\\.\\.\\.\\a^{j}\\ \end{bmatrix}$
  $a^{l}=\begin{bmatrix}a_{1}\\.\\.\\.\\a^{k}\\ \end{bmatrix}$
  $Y^{m}=\begin{bmatrix}y^{m}\end{bmatrix}$
* $W^{l}=\begin{bmatrix}w_{11}&...&w_{1k}\\&.\\&.\\&.\\w_{j1}&...&w_{jk}\\ \end{bmatrix}$
  
* $$a^{l}=W^{l}_{transpose}*a^{l-1}$$
### 6.역전파
* 신경망 모델은 미분을 통해 학습하는데 이는 역전파를 통해 구현할 수 있다.  
  역전파를 통하여 최초 입력 값(X)이 최종 결과 값(L)에 미치는 영향을 구할 수 있다.
*  $${dL\over dX }={dX\over da^{1}}*{da^{1}\over da^{2}}*...*{dL\over da^{L}}$$

### 7.신경망의 정전파와 역전파
* Forward propagation : 입력 값에 대한 예측 값을 구하는 정방향 계산이다.   
  input: $a^{l-1}$ output: $a^{l}$  
  $z^{l}=w^{l}*a^{l-1}+b^{l}$  
  $a^{l}=g^{l}(z^{l})$
* Back propagation : 경사 하강법을 위한 도함수를 계산하는 역방향 계산이다.  
  input: $da^{l}$, $z^{l}$ output: $da^{l-1}$  
  최초 입력 값까지 chain rule 반복 실행