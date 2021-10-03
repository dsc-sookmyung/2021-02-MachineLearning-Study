# 5. 심층 신경망 네트워크

Deep L-layer Neural network
---
![](https://user-images.githubusercontent.com/90624848/135772252-7ebd4835-7797-49f3-9b1c-9eaf5279f1dd.PNG)

![enter image description here](https://user-images.githubusercontent.com/90624848/135772313-8b6aecaa-e113-443a-aebb-fbcf1596ed62.PNG)
![](https://user-images.githubusercontent.com/90624848/135772348-65b016e8-ad36-4650-8291-4ffc868c7f19.PNG)
- L = 4
- n^[1] = 5, n^[2] = 5, n^[3] = 3, n^[4] = 1 = n^[L]

Forward and Backward Propagation
---
**Forward propagation for layer l**
![](https://user-images.githubusercontent.com/90624848/135772541-d222286f-465b-4111-8a2d-aebc56f48962.PNG)
- a^[0]: 한 번에 하나씩 할 경우의 학습 데이터에 대한 입력 특성
- A^[0]: 전체 학습 세트를 진행할 때의 입력 특성
⇒ 이 과정을 반복하는 것: 정방향 전파 계산

**Backward propagation for layer l**
![](https://user-images.githubusercontent.com/90624848/135772577-b258f53b-7023-47e6-a6fd-51fa174a71f3.PNG)
![](https://user-images.githubusercontent.com/90624848/135772645-19b667a2-1b7b-4a34-b6be-e062db48d88d.PNG)

**3개의 층을 가진 심층 신경망에서 정방향 전파와 역방향 전파를  구현하는 방법**
![](https://user-images.githubusercontent.com/90624848/135772678-7ebc4ad6-2def-4603-be73-961fb4bb5447.PNG)
- 정방향 반복: 입력 데이터 X로 초기화
- 역방향 반복: da^[l] = L을 미분한 값

Forward Propagation in a Deep Network
---
![](https://user-images.githubusercontent.com/90624848/135772911-0799691e-847e-4476-a90a-ec0ae5b58a24.PNG)
- 1부터 L까지 심층 신경망의 모든 층에 대해 활성화를 게산하는 반복문은 명시적으로 써도 됨.

Getting Matrix Dimensions Right
---
![](https://user-images.githubusercontent.com/90624848/135772954-9849bb8a-946a-4c57-9260-0c67fd7903bf.PNG)
![](https://user-images.githubusercontent.com/90624848/135772978-de4adebf-d2e7-4d26-8430-1fa939877579.PNG)

Why Deep Representations?
---
- 깊은 심층 신경망이 더 잘 작동하는 직관적인 이유
	1. 낮은층에서는 간단한 특징을 찾아내고,  
	그 후 깊은 층에서는 탐지된 간단한 것들을 함께 모아서 더 복잡한 특징을 포착함.  
	⇒ 초기의 층: 간단한 함수, 깊어질수록 복잡해짐
	2. 순환 이론
	![enter image description here](https://user-images.githubusercontent.com/90624848/135773223-369d2512-f856-4999-8997-5c1eb81b33a6.PNG)
		- 은닉수가 작은 깊은 심층망에서 계산할 수 있어도 얕은 네트워크로 계산하려고 하면(충분한 은닉층 X) 많은 은닉 유닛이 계산에 필요


Building Blocks of a Deep Neural Network
---
**Forward and Backward functions**
![enter image description here](https://user-images.githubusercontent.com/90624848/135773267-2b682abf-77fb-44fa-8083-1f3bea6c33c8.PNG)
![](https://user-images.githubusercontent.com/90624848/135773314-24e3eb5a-f933-45d3-85f7-6b071ed93cb6.PNG)
- 구현의 관점에서 매개변수들의 값을 저장하고 나중에 있을 역전파의 게산에서 필요한 곳에 복사해서 사용하는 것이 유용함.



Parameters vs Hyperparameters
---
- Parameters : 신경망에서 학습 가능한 W와 b
- Hyperparameters: 학습 알고리즘에 알려줘야 하는 것들
	- 종류
		- 학습률(learning rate, α)
		- 반복횟수(# of iteration)
		- 은닉층의 개수(# of ihidden layer, L)
		- 은닉유닛의 개수 (# of hidden units)
		- 활성화 함수의 선택(choice of activation function)
		- 모멘텀항(momentum term)
		- 미니배치 크기(mini batch size)  
	- 궁극적으로 최종 매개변수인 W와 b를 통제함(최종 값을 결정하기 때문)
	- 결정된 것이 없으며, 여러번의 시도를 통해 적합한 하이퍼파라미터를 찾아야함
