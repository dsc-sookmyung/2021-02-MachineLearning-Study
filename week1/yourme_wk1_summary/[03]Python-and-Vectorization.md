# 3. 파이썬과 벡터화

Vectorization
---
- 딥러닝: 빠르게 코드가 처리되어야 함.
+ SIMD(Single Instruction Multiple Data)
	- 병렬 프로세서의 한 종류
	- 하나의 명령어로 여러 개의 값을 동시에 계산하는 방식
	⇒ 벡터화 연산 가능
	$\therefore$ 벡터로 만들어서 한번에 연산하는 것이 더 효율적

**Neural network programming guideline**
- 신경망이나 로지스틱 회귀를 프로그램밍 할 때 기억할 것
- Whenever possible, **avoid explicit for-loops**

```numpy
# vector v에 대한 연산

# none-vectorization(use for-loops)
u = np.zeros((n,1))
for i in range(n):
	u[i]=math.exp(v[i])

# use numpy func.
import numpy as np
u = np.exp(v)
```


Vectorizing Logistic Regression
---
**정방향 전파하는 벡터화된 구현**
+ for문을 이용해 i의 값을 변화시키며 계산해야 하는 식
	- $z^{(i)}=W^Tx^{(i)}+b$
	- $a^{(i)}=\sigma (z^{(i)})$
	
	⇒ 벡터화 ``Z = np.dot(ntranspose(Wt,X) +b``
	　　　　  $dw_1, dw_2$ ⇒ $dw = dw + x^{(i)}\ast dz^{(i)}$
+ z와 a를 하나씩 계산하기 위해 m개의 훈련 샘플을 순환하는 대신 한 줄의 코드로 모든 z를 동시에 계산
+ 적절한 $\sigma$의 구현으로 한 줄의 코드로 모든 a를 동시에 계산 ⇒ $\sigma (Z)$




Vectorizing Logistic Regression's Gradient Computation
---
+ $dz^{(i)}=a^{(i)}-y^{(i)}$
+ for문 사용
	![for-loops](https://cphinf.pstatic.net/mooc/20180615_244/1529028895251WEzUQ_PNG/pic1.PNG)

+ 벡터 사용
![vectorization](https://cphinf.pstatic.net/mooc/20180615_8/152902893889842v4T_PNG/pic2.PNG)


Broadcasting in Python
---
한 줄의 파이썬 코드만으로 각 열의 합 구하기
```
cal = A.sum(axis=0)	# 각 열의 합
percentage = 100 * A/cal.reshape(1,4)
```
- ``axis = 0`` : 세로축  
	``axis = 1`` : 가로축
- ``percentage = 100 * A/cal.reshape(1,4)`` : (3,4) 행렬인 A를 (1,4) 행렬로 나누는 것
	``cal``은 이미 (1,4) 행렬이므로 ``reshape`` 사용할 필요 X
	but 행렬의 차원이 확실하지 않다면 ``reshape``를 사용하여 필요한 차원의 행렬로 확실히 만들어 줌
$\because$ ``reshape`` 함수는 상수 시간이 걸림 ⇒ 호출이 저렴함
- (m, n) + (1, n) ⇒ (m, n) + (**m**, n)
	(m, n) + (m, 1) ⇒ (m, n) + (m, **n**)
	- 파이썬: (1, n) ⇒ (m, n)으로 만들어줌.
	- if (2,3) + (1,3) ⇒ (2,3) + (2,3)

A Note on Python/Numpy Vectors
---
**코드 작성 팁**
- Don't use "rank 1" array ⇒ 행 or 열 벡터로 만들것
	``a = np.random.randn(5)``
	⇒ ``a = np.random.randn(5,1)`` 열 벡터
	⇒ ``a = np.random.randn(1,5)`` 행 벡터
- ``assert(a.shape== (5,1)`` : 행렬과 배열의 차원 확인
- ``a = a.reshape((5,1))`` : 행렬과 벡터를 필요한 차원으로 만듦


Explanation of Logistic Regression's Cost Function
---
- **손실함수**
	- y값이 1이 될 확률: $P(y=1|x) = \hat{y}$
	- y값이 0이 될 확률: $P(y=0|x) = 1 - \hat{y}$  
	⇒ $P(y|x) = \hat{y}^y(1-\hat{y})^{(1-y)}$  
	⇒ $\log P(y|x) = log(\hat{y}^y(1-\hat{y})^{(1-y)}) = y\log \hat{y}+(1-y) \log (1-\hat{y})$
	- 목적: 확률($\log P(y|x)$)을 최대화 시키는 것
	⇒ 동치인 -1을 곱한 확률($-\log P(y|x)$) 최소화를 목적으로 손실함수 정의.
	손실 함수를 최소화 시키는 것 == 확률의 로그값을 최대화 시키는 것
	$\therefore L(\hat{y},y)=-\log P(y|x) = -(\hat{y}^y(1-\hat{y})^{(1-y)})$
- **비용함수**
	- m개의 훈련 세트 중, 각 샘플($x^{(i)}$)이 주어졌을 때,  
	샘플에 해당하는 라벨($y^{(i)}$)값이 1 혹은 0이 될 확률의 곱
	- 훈련 세트의 타깃 확률을 **최대화**해주는 변수를 찾아야 함.
	- $P(labels \ in \ training \ set) = \prod_{(i=1)}^mP(y^{(i)}|x^{(i)})$
	⇒ $\log P(labels \ in training \ set) = \log \prod_{(i=1)}^mP(y^{(i)}|x^{(i)}) = - \sum_{(i=1)}^mL(\hat{y},y)$ 
	$\therefore J(w,b) = -\log P(labels \ in \ training \ set) = {1 \over m}\sum_{(i=1)}^mL(\hat{y},y)$
	- 비용함수인 $J(w,b)$를 최소화하므로 로지스틱 회귀 모델의 Maximum likelihood estimation을 한 것
