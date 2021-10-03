# 1. 딥러닝 소개

What is a Neural Network?
---
  

- 딥러닝: 신경망을 학습시키는 것
+ 신경망
	- 구성: 입력층 - 은닉층 - 출력층
	- 은닉층: 입력층의모든 입력을 다 받음
	- 입력층 - 은닉층: 조밀하게 연결되어있음
		∵ 모든 입력 특성들은 중앙에 있는 원 모두에 연결되어 있기 때문이다.
- 충분한 양의 x와 y를 훈련 예제로 주면 정확도 높아짐

Supervised Learning with a Neural Network
---
- **지도학습(Supervised Learning)**: 입력 x와 출력 y에 매핑되는 함수를 학습하려 함

| Input(x) | Output(y) | Application | NN Model
|--|--|--|--|
| Home features | Price | Real Estate | Standard NN |
| Ad, user info | Click on ad? (0/1) | Online Advertising | Standard NN |
| Image | Object(1,…,1000) | Photo tagging | CNN |
| Audio | Text transcript | Speech recognition | RNN |
| English | Chinese | Machine translation | RNN |
| Image, Radar info | Position of other cars | Autonomous driving | Custom/Hybrid |


+ Neural Network examples
	- CNN: 이미지 데이터
	- RNN: 1차원 시퀀스 데이터에 강함
+ Structured Data vs Unstructured Data
	- Structured Data: 데이터베이스로 표현된 데이터
	- Size, #bedrooms처럼 특성들이 잘 정의되어 있음
+ **Unstructured Data**
	- 음성파일, 이미지, 텍스트 데이터
	- 특성: 이미지의 픽셀값/텍스트의 각 단어 같은 것

Why is Deep Learning taking off?
---
+  **Scale drives deep learning progress**
	- Traditional learning algorithm: 어느정도 지나면 성능이 정체기에 이르게 됨 > 방대한 양의 데이터를 활용X
	- 신경망의 크기가 커질수록 성능은 계속해서 좋아지고 있음.
		* 성능을 발휘하기 위한 조건
			1. 많은 양의 데이터를 이용하기 위해 충분히 큰 신경망이 필요함
			2. 많은 양의 데이터가 필요함
　 >> 보통 규모가 딥러닝의 발전을 주도했다고 말함.
	- Amount of (**labeled**) data (=m) = (x,y)가 같이 있는 데이터
	- 적은 훈련 세트가 있는 경우 알고리즘의 상대적 순위가 잘 정의도어 있지 않음
∴ 구현방법에 따라 성능이 결정되는 경우 多
　>> SVM을 훈련시키는 데 여러 특성을 잘 관리한다면 더 큰 신경망보다 SVM이 나을수도 있음
	- m이 아주 클 때에만 큰 신경망이 일관되게 다른 신경망을 압도함

- 최근 딥러닝이 강력한 도구로 부상한 이유
	1. data: 데이터 양 증가
	2. computation: 컴퓨터 성능 향상
	3. Algorithms: 알고리즘의 개선
		- ex) Sigmoid 함수가 아닌 ReLU 함수를 사용함으로써 Gradient 소멸 문제 해결

			**Sigmoid 함수와 ReLU 함수**
sigmoid는 왼쪽, 오른쪽 끝으로 가면 미분값이 0이 되기 때문에 Gradient 가 소멸하는 문제가 발생하는데, ReLU 함수를 사용하므로 문제를 해결할 수 있음.
