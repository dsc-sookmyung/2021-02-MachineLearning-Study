# 머신러닝 2주차
## 머신러닝 어플리케이션 설정
### 1. 모델 학습 과정
* iterative process : 모델의 성능 향상을 위해 # of layers, # of units, learning rate, activation function을 최적화 한다.
* data set 설정: 최적화를 위한 iterative process를 효율적으로 진행 하는데 도움이 된다.
* data set = train set : dev set : test set  
 train set:모델 학습용   
 Dev set: 학습된 모델들 성능 평가  
 Test set: 선택된 모델의 작동 확인  
 소규모 데이터: 7:3 또는 6:2;2  
 대규모 데이터: 98:1:1 또는 99.5:0.4:0.1
## 모델 성능 분석
### 1. bias / variance
* train set error: 오류 값이 클 수록 high bias
* dev set error: training error와 차이가 클 수록 high variance
### 2. error에 따른 해결 방법
* traning error (high bias) : # of layers, # of units 증가,epoch 증가, 데이터 화질 개선
* dev error (high variance) : training set 증가, 데이터 정규화
### 3. dev error 줄이기
* dev error (high varinace)를 줄이기 위해 데이터 정규화를 이용한다.
* L2 :  
   $$J(W,b)=\frac{1}{m}\sum_{i=1}^{m} L(y_{predict},y_{real}) + \frac{2\lambda}{m}\sum_{j=1}^{nx} w_{j}^{2}$$
     $$w^{l}:=(1-\frac{\alpha\lambda}{m})w^{l}-\alpha$$
   이 때 가중치 매개변수 $\lambda$가 가중치 $W$를 감쇠 시킨다.  
   가중치가 감쇠하면 모델 단순화(linear)로 overfitting을 방지한다.
* Drop out: 네트워크 각 층의 노드를 삭제할 확률을 지정한다.  
  무작위로 삭제된 노드에 의해 특정 입력이 특정 노드에 대한 의존도를 낮추고 네트워크가 간소화 된다. 
* 다른 방법: 데이터 증식(rotate, crop 등)으로 traing set 증가, early stop으로 과적합 되기 전에 학습 종료
## Multiclassification
### 1. softmax regression
* multi classfication의 마지막 층의 activation function으로 선형 값을 확률 값으로 변환한다.
* softmax regression:   
  $$a_{i}=\frac{e_{zi}}{\sum_{j=1}^{C}e_{zj}}$$
* multi classification의 Loss function:   
  $$L(y_{predict},y_{real})=\sum_{j=1}^{N}-y_{j}log(y_{j})$$
### 2. convolution Neural Network
* convolution 연산  
  featuer extraction: 특정 feature에 대응하는 filter와 convolution 연산 시 값이 클수록 해당 feature와 유사  
  윤곽선 추출 filter:  
  세로 윤곽선 filter : $\begin{bmatrix}1 & 0 &-1\\1 & 0 &-1\\1 & 0 &-1\\\end{bmatrix}$ 
  가로 윤곽선 filter : $\begin{bmatrix}1 & 1 & 1\\0& 0 &0\\-1 & -1 &-1\\\end{bmatrix}$   
  최근 filter: 임의의 수로 초기화 후 역전파로 학습
* filter,stride와padding  
  filter size와 stride 에 따라 output data의 크기가 달라진다. n -> n-f+1  
  이 때 convoultion을 반복 할수록 데이터 크기가 축소되어 윤곽선 정보가 유실된다.  
  이를 방지하기 위해 padding을 적용한다. n -> n-f+1+2p
* pooling    
  계산속도를 줄이고 특징을 더 잘 검출 한다.
  max pooling: 필터 부분 중 가장 큰 값을 다음 output 값으로 지정  
  average pooling: 필터 부분의 평균 값을 다음 output 값으로 지정  
  convolutiond 연산은 특징과 유사 할 수록 값이 크므로 보통 maxpooling을 사용함
* LeNet-5  
``` python
input_=tf.keras.Input((32,32,1))      # 28 > 32는 padding 최소화
x=tf.keras.layers.Conv2D(6,5)(input_) # 5by5는 큰 filter 3by3이 보편적
x=tf.keras.layers.Activation('tanh')(x)  
x = tf.keras.layers.AvgPool2D(2,2)(x) # 반으로 줄임 non-overlapping
x =tf.keras.layers.Conv2D(16,5)(x)    # 가장자리에 중요 특징 padding='same'
x=tf.keras.layers.Activation('tanh')(x)   
x = tf.keras.layers.AvgPool2D(2,2)(x) # feature extraction
x= tf.keras.layers.Flatten()(x)       # tf.keras.layers.GlobalAveragePooling2D()(x)
x= tf.keras.layers.Dense(120)(x)
x= tf.keras.layers.Dense(84)(x)
x= tf.keras.layers.Dense(10)(x)       # fully connected

```  
``` python  
LeNet-5
> Convolution과 전통적인 mL 기법을 결합한 CNN 모델
> activation fucntion으로 tanh 사용
> gradient vanishing으로 deeplearning 불가
```




