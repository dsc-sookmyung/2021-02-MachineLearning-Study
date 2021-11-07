# 파이토치와 신경망 기초

구글 코랩(colab) : 웹브라우저 만으로, 구글의 인프라에서 파이썬 코드가 돌아가도록 지원한다.

파이토치 : 복잡한 신경망을 만드는데 필요한 많은 작업을 대신함으로 우리가 신경망을 설게하는 일에 집중할 수 있도록 해준다

## 파이토치 사용
파이토치를 사용하려면 torch모듈을 임포트해야 한다

import torch
파이토치의 변수 : tensor

x = torch.tensor(3.5, requires_grad = True)
y = (x-1) * (x-2) * (x-3)
일반적인 파이썬 변수, 넘파이 행렬과 달리 파이토치 변수인 텐서는 y가 x로부터 만들어졌고, 어떻게 계산되었는지를 기억한다.
requires_grad = True 옵션은 파이토치에게 x에 대한 기울기 계산이 가능하도록 하는 기능을 제공
다차원 행렬, 2차우너 행렬, 1차원 리스트, 단일 값 등이 가능
기울기 계산(미분)

y.backward()
신경망을 훈련시키기 위해서는 미분을 위한 기울기의 오차와 가중치 변화에 따른 결과와의 오차 변화를 알아야 한다.
위 식을 통해 y가 (x-1) x (x-2) x (x-2)이라는 수식에 의해 계산된다는 사실을 이용해 자동으로 dy/dx를 계산해 나온 기울기를 텐서 x에 저장한다.
print(x.grad)#기울기 확인
x → y → z로 구성된 신경망인 경우
x = torch.tensor(3.5, requires_grad = True)
y = x*x
z = 2*y + 3

z.backward() #기울기 계산

print(x.grad)​
파이토치는 정방향의 계산그래프를 만든 후 backward()함수를통해 역방향으로도 작동하도록 기울기 dz/dx를 계산해 텐서 x 안의 x.grad에 저장한다
a, b → x, y → z로 구성된 신경망인 경우
a = torch.tensor(2.0, requires_grad = True)
b = torch.tensor(1.0, requires_grad = True)

x = 2*a + 3*b
y = 5*a*a + 3*b*b*b
z = 2*x + 3*y

z.backward()

a.grad​
결과가 z, 올바른 결과가 t로 표현된다고 할 때 오차 E는 (z-t)^2
즉, 오차 E는 사실 z에서 E로 계산되는, 마지막 노드가 (z-t)^2인 네트워크의 한 노드가 된다
파이토치는 네트워크 각각의 입력을 통해 E에 대한 경사를 구할 수 있다
신경망을 훈련시키기 위해 dE/dw(w는 네트워크 내의 가중치)를 계산하지만, 가중치도 노드의 일부라고 생각하면 결국 dz/da와 다름이 없다

---

## 신경망 만들기
### MNIST 이미지 데이터 셋

머신러닝 알고리즘을 테스트하는데 널리 쓰이는 유명한 데이터 셋
훈련용 60,000개, 테스트용 10,000개의 손글씨 이미지가 들어잇다
각 이미지는 단일 색상의 28 x 28 픽셀이며, 각 픽셀은 0~255사이의 값을 갖는다

### 데이터 살펴보기

구글 드라이브에 업로드한 데이터를 파이썬에서 접근할 수 있도록 드라이브 마운트
from google.colab import drive
drive.mount('./mount')
실행 결과의 링크를 클릭해 접근 권한을 확인한 후 인증코드를 노트북 결과에 다시 붙여넣고 <enter>키를 누르면 ./mount를 통해 파일에 접근할 수 있게 된다
판다스 라이브러리를 통해 훈련용 데이터를 데이터 프레임으로 불러온다
import pandas
df = pandas.read_csv('mount/My Drive/Colab Notebooks/myo_gan/mnist_data/mnist_train.csv', header=None)
데이터 프레임은 넘파이 행렬과 비슷한 기능을 하나, 열과 행의 관리 등 추가적인 기능이 제공되며, 합계를 구하거나 데이터를 필터링하는 등의 편리한 기능도 제공된다
맷플롯립 라이브러리를 통해 pyplot을 사용해 이미지를 확인할 수 있도록 한다
import matplotlib.pyplot as plt

row = 0
data = df.iloc[row]

#첫번째 값은 레이블
label = data[0]

#이미지 데이터는 나머지 784개의 값
img = data[1:].values.reshape(28, 28)
plt.title("label = " + str(label))
plt.imshow(img, interpolation='none', cmap = 'Blues')
plt.show()

### 간단한 신경망

신경망 가장 처음에 있는 것은 MNIST이미지
이미지는 28 x 28, 즉 784개의 픽셀 값으로 이루어져 있다
이는 곧 신경망의 맨 처음 레이어(입력레이어, input 레이어)가 784개 노드를 가지고 있다는 의미
한 레이어의 모든 노드가 다음 레이어의 모든 노드와 연결이 되어있는 경우 완전 연결이라고 부른다
한 레이어에서 다음 레이어로 이동할 때, 출력에이어와 은닉 레이어의 출력부분에 붙어있는 활성화 함수로 어떤 것을 사용할지 결정해야 한다
어떤 신경망을 만들든 항상 파이토치의 torch.nn을 상속받아 클래스를 만들어야 한다.
이를 상속받으면 자연스럽게 파이토치는 계산 그래프를 만들고, 훈련 시 가중치를 조정하는 과정을 진행한다

### 분류기 만들기
	
import torch
import torch.nn as nn #torch.nn모듈은 nn이라는 이름으로 임포트하는 것이 관습

class Classifier(nn.Module):
  def __init__(self):
    super().__init__()

    self.model = nn.Sequential(
        nn.Linear(784, 200), #784개의 노드로부터 200개의 노드까지 완전 연결 매핑
        nn.Sigmoid(), #S모양의 로지스틱 활성화 함수로 이전레이어부터의 출력에 적용
        nn.Linear(200,10), #200개의 노드로부터 10개의 노드로 완전 연결 매칭
        nn.Sigmoid() #신경망의 최종 출력
    )

    self.loss_function = nn.MSELoss()#평균제곱오차법 : 실제와 예측된 결과 사이의 차이의 제곱의 평균 계산 , 학습 파라미터 업데이트

    self.optimiser = torch.optim.SGD(self.parameters(), lr = 0.01)#확률적 경사하강법 , 학습 파라미터 업데이트

    self.counter = 0
    self.progress = []

    pass
  
  def forward(self, inputs):
    return self.model(inputs)

  def train(self, inputs, targets):
    outputs = self.forward(inputs)

    loss = self.loss_function(outputs, targets)

    #각 노드마다 오차기울기를 계산하고, 노드에 연결된 가중치 수정
    self.optimiser.zero_grad()#기울기 초기화
    loss.backward()#역전파 실행
    self.optimiser.step()#가중치 갱신

    self.counter += 1
    if(self.counter%10 == 0):
      self.progress.append(loss.item())
      pass

    if(self.counter %10000 == 0):
      print("counter = ", self.counter)
      pass

  def plot_progress(self): #훈련 시각화
    df = pandas.DataFrame(self.progress, columns = ['loss'])#차트로 쉽게 나타내기 위해 손실값을 저장해둔 리스트를 팬더스 데이터프레임으로 변환
    df.plot(ylim = (0, 1.0), figsize = (16, 8), alpha = 0.1, marker='.', grid = True, yticks = (0, 0.25, 0.5))#plot함수의 옵션으로 여러 스타일과 디자인 지정
    pass

학습 파라미터 : nn.Linear()을 통해 Ax + B의 선형형태로 출력값이 다음 레이어로 전달된다. A와 B를 학습 파라미터라고 부른다. 여기서 A는 가중치, B는 편향을 의미하고, 두 파리미터 모두 훈련 중에 업데이트 된다.

### MNIST데이터셋 클래스

from torch.utils.data import Dataset

class MnistDataset(Dataset):
  def __init__(self, csv_file):
    self.data_df = pandas.read_csv(csv_file, header = None)
    pass

  def __len__(self): #데이터 셋의 길이를 반환
    return len(self.data_df)

  def __getitem__(self, index): #데이터셋의 n번째 아이템을 반환
    label = self.data_df.iloc[index, 0]
    target = torch.zeros((10))
    target[label] = 1.0

    image_values = torch.FloatTensor(self.data_df.iloc[index, 1:].values) / 255 #각 픽셀을 255로 나눔 -> 정규화, 0~1사이의 값을 갖는다
    
    return label, image_values, target

  def plot_image(self, index):
    img = self.data_df.iloc[index,1:].values.reshape(28, 28)
    plt.title("label = " + str(self.data_df.iloc[index, 0]))
    plt.imshow(img, interpolation = 'none', cmap = "Blues")
    #plt.show()
    pass

  pass
원핫 인코딩 : 길이가 10이고 정답만 1로 표시된 텐서, [0, 0, 1, 0, 0, 0, 0, 0, 0, 0] 이러한 방식을 의미
분류기 훈련시키기

### 신경망 생성 및 훈련
C = Classifier()

%%time
C = Classifier()

epochs = 3

for i in range(epochs):
  print("training epoch =", i+1, "of", epochs)
  for label, image_data_tensor, target_tensor in mnist_dataset:
    C.train(image_data_tensor, target_tensor)
    pass
  pass
오차 출력
C.plot_progress()

### 신경망에 쿼리하기

이미지를 실제로 분류할 수 있는지 시도하기
MNIST 테스트용 10,000개 이미지를 대상으로 성능 파악하기
mnist_test_dataset = MnistDataset('mount/My Drive/Colab Notebooks/myo_gan/mnist_data/mnist_test.csv') #MNIST 테스트 데이터 로드

record = 42

mnist_test_dataset.plot_image(record)

image_data = mnist_test_dataset[record][1]

output = C.forward(image_data)

pandas.DataFrame(output.detach().numpy()).plot(kind = 'bar', legend = False, ylim=(0, 1))

score = 0
items = 0

for label, image_data_tensor, target_tensor in mnist_test_dataset:
  answer = C.forward(image_data_tensor).detach().numpy()
  if(answer.argmax() == label):
    score += 1
    pass
  items += 1

  pass

print(score, items, score/items) #분류기 성능 출력

---
	
##성능 향상 기법
### 손실 함수

연속적인 숫자를 맞춰야 하는 경우(Ex. 섭씨온도를 맞추는 모델) → 평균제곱오차법(MSE) → 회귀문제
참/거짓 혹은 1/0과 같은 이산형 결과를 맞춰야 하는 경우 (Ex. 고양이인지 아닌지 맞추는 모델) → 이진교차엔트로피(BCE)손실 → 분류문제
확실하게 틀리는 경우에 특히 큰 페널티를 주는 방식으로, 결과 노드는 0.0에 가깝거나 1.0에 가깝게 판단을 한다.
self.loss_function = nn.BCELoss()

차트는 느리게 하강하고 노이즈가 많지만, 훈련 후반부로 갈수록 손실이 더 작아 더 나은 결과를 도출한다.

### 활성화 함수

S모양의 로지스틱 함수는 신경망 초기에 많이 사용했다.
동물 뉴런에서 일어나는 신호 전달 현상과 비슷하고, 기울기 계산이 수학적으로 간편하기 때문
그러나 큰 값들에 대해 기울기가 굉장히 작아지다가 결과적으로 사라질 수 있다.(포화상태)
로지스틱 함수의 문제를 해결하기 위해 ReLU(정류선형유닛)을 사용하게 되었다.
ReLU의 경우 0보다 작은 값들에 대해서는 경사가 0이기 때문에 기울기가 소실되는 문제가 존재한다.
ReLU를 약간 수정해 0보다 작은 경우에 미세한 기울기를 허용하는 리키(Leaky) ReLU를 사용하기도 한다.


손실은 0으로 빠르게 떨어지고, 모델 훈련의 아주 초반에도 손실이 낮고, 노이즈도 적다.

### 옵티마이저
역전파시 가중치 업데이트 방식
확률적 경사하강법
	국소 최적해에 빠질 확률이 높다.
 	모든 파라미터에 단일한 학습률을 적용한다.
Adam
	관성을 이용해 국소 최적해에 빠질 가능성을 줄였다.
	학습 파라미터에 대해 각각 다른 학습률을 적용했고, 학습시 파라미터를 상황에 따라 수정한다.
	self.optimiser = torch.optim.Adam(self.parameters())

손실이 빠르게 0으로 떨어지며, 평균또한 굉장히 낮다.
완벽하진 않아도 많은 상황에서 Adam은 좋은 선택이다.

### 정규화
신경망의 가중치와 여러 신호의 값이 굉장히 큰 값을 가질 때가 존재한다. 이런 경우 중요한 값이 소실되는 결과가 나올 수 있고, 훈련을 어렵게 만들 수 있다.
정규화
신경망 훈련을 안정화 시킨다.
파라미터의 범위를 감소시키거나, 평균을 0으로 맞춰주는 작업이 상당히 도움이 된다는 많은 연구결과가 존재한다.
self.model = nn.Sequaltial(
	nn.Linear(784, 200),
    nn.LeakyReLU(0.02),
    
    nn.LayerNorm(200),
    
    nn.Linear(200, 10),
    nn.LeakyReLU(0.02)
)

원래 모델보다 손실이 굉장히 빠르게 감소하고, 손실의 변동폭도 상당히 작은 편이다.
위의 모든 방법을 종합하여 이용할 수 있다.

---	
	
## CUDA
GPU(그래픽 처리 장치)는 CPU와는 다르게 특정한 작업만 잘 하도록 설계되어 있다.
컴퓨터의 특정한 하드웨어로부터 좀 더 나은 성능을 얻기 위한 연구의 결과, 그래픽 성능을 높이기 위한 하드웨어인 그래픽카드를 이용하는 것이 도움이 된다는 것을 알게 되었다.
CPU의 경우 : 아무리 코어 수가 많아도 64코어정도가 최대로, 하나하나 연산을 수행할 것
GPU의 경우 : 천 개 이상의 코어가 하나의 GPU안에 들어있는 것이 일반적이다 = 굉장히 많은 연산이 쪼개져서 병렬로 계산될 수 있다
NVIDIA는 GPU시장의 리더역할을 수행하고 있는 기업으로, 강력한 하드웨어 가속기능을 갖춘 소프트웨어 프레임워크 CUDA를 제공한다
Colab에서 CUDA이용하기
노트북 메뉴 상단의 '런타임' → '런타임 유형 변경' 을 선택하여 하드웨어 가속기를 'None'에서 'GPU'로 바꿔준다
GPU에서 텐서를 만들고 싶으면 타입으로 torch.cuda.FloatTensor를 이용한다
x = torch.cuda.FloatTensor([3.5])
x.device를 통해 어떤 장치에 텐서가 올라가 있는지 확인할 수 있다
x.device
노트북 처음에 CUDA를 사용하게끔 선언하고, 장치를 확인하는 코드
if torch.cuda.is_available():
	torch.set_default_tensor_type(torch.cuda.FloatTensor)
    print("using cuda : ", torch.cuda.get_device_name(0))
    pass
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

device
