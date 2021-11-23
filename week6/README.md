조건부 GAN
---

### **✨ 목표**
이미지를 **단일한 클래스로 고정한 채로 다양한 이미지를 생성**할 수 있게 하는 것  

🙋‍♀️ 개발자: 숫자 3을 표현하는 다양한 이미지를 생성해줘!   
💻 GAN: OK!   

### **💡 조건부 GAN 구조**
![image](https://user-images.githubusercontent.com/90624848/142963347-73857158-54e8-4673-a8fd-c65bff5ffd72.png)   
🌟 판별기와 생성기 모두 이미지 데이터 외에도 클래스 레이블을 추가로 입력받는다.   
1. 생성기에 임의의 시드와 함께 어떤 이미지를 원하는지 입력을 넣어주어야 한다.
2. 판별기는 클래스 레이블과 이미지 사이의 관계를 학습해야하므로,   
판별기에도 클래스 레이블에 대한 정보를 같이 제공해야한다.

### **🔎 주요 코드 살펴보기**
#### ✔ 판별기   
이미지 픽셀 데이터와 클래스 레이블 정보를 동시에 받도록 판별기를 업데이트 해야함.   
방법 : ```forward()``` 함수에서 이미지 텐서와 레이블 텐서를 동시에 받게 하고 결합한다.   

```forward()```: 시드와 레이블을 결합한다.   
```python
def forward(self, image_tensor, label_tensor):
        # 시드와 레이블 결합
        inputs = torch.cat((image_tensor, label_tensor))
        return self.model(inputs)
```

```self.model```: 두 텐서를 이은 길이 = 784(이미지 텐서의 길이) + 10(레이블 텐서의 길이)
```python
self.model = nn.Sequential(
    nn.Linear(784+10, 200),
    nn.LeakyReLU(0.02),

    nn.LayerNorm(200),

    nn.Linear(200, 1),
    nn.Sigmoid()
)
```

```train()```: forward()를 호출할 때 레이블을 추가한다.
```python
def train(self, inputs, label_tensor, targets):
      # 신경망 출력 계산
      # forward()를 호출할 때 레이블 추가
      outputs = self.forward(inputs, label_tensor)

      loss = self.loss_function(outputs, targets)

      self.counter += 1;
      if (self.counter % 10 == 0):
          self.progress.append(loss.item())
          pass
      if (self.counter % 10000 == 0):
          print("counter = ", self.counter)
          pass

      self.optimiser.zero_grad()
      loss.backward()
      self.optimiser.step()

      pass
```

```훈련 반복문```: 레이블 텐서를 추가로 train() 함수에 전달한다.
```python
for label, image_data_tensor, label_tensor in mnist_dataset:
    # 실제 데이터
    D.train(image_data_tensor, label_tensor, torch.FloatTensor([1.0]))
    # 생성 데이터
    D.train(generate_random_image(784), generate_random_one_hot(10), torch.FloatTensor([0.0]))
    pass
```

```generate_random_one_hot()```: 임의의 원핫 인코딩된 클래스 레이블 벡터, 크기는 정수로 지정해야한다.
```python
def generate_random_one_hot(size):
    label_tensor = torch.zeros((size))
    random_idx = random.randint(0,size-1)
    label_tensor[random_idx] = 1.0
    return label_tensor
```

#### ✔ 생성기   
시드와 레이블 텐서를 생성기에 투입하게 했으므로, 두 텐서를 결함해서 신경망에 전달하게 수정해야 한다.

```forward()```: 두 텐서를 결합해서 신경망에 전달
```python
def forward(self, seed_tensor, label_tensor):        
    # 시드와 레이블 결합
    inputs = torch.cat((seed_tensor, label_tensor))
    return self.model(inputs)
```

```self.model```: 네트워크의 첫 번재 레이어는 10개의 추가적인 값을 다루어야 한다.
```python
self.model = nn.Sequential(
            nn.Linear(100+10, 200),
            nn.LeakyReLU(0.02),

            nn.LayerNorm(200),

            nn.Linear(200, 784),
            nn.Sigmoid()
        )
```

```train()```: 레이블 텐서를 받도록 수정한다.
```python
def train(self, D, inputs, label_tensor, targets):
      # 신경망 출력 계산
      g_output = self.forward(inputs, label_tensor)

      # 판별기로 전달
      # 생성기에서 생성된 이미지들을 판별기의 forward()함수에 넘김
      # --> 생성기가 다른 레이블로 잘못 판단하는 것을 방지
      d_output = D.forward(g_output, label_tensor)

      loss = D.loss_function(d_output, targets)

      self.counter += 1;
      if (self.counter % 10 == 0):
          self.progress.append(loss.item())
          pass

      self.optimiser.zero_grad()
      loss.backward()
      self.optimiser.step()

      pass
```

```plot_images()```: 차트 그리기   
label을 정수로 받아서, 이로부터 원핫 인코딩된 텐서를 만들고 생성기에 전달한다.  
여섯 개의 다른 임의의 시드로 여섯 개의 이미지가 생성되어 최종적으로 격자에 그려진다.
```python
def plot_images(self, label):
    label_tensor = torch.zeros((10))
    label_tensor[label] = 1.0
    # 2행 3열로 샘플 이미지 출력
    f, axarr = plt.subplots(2,3, figsize=(16,8))
    for i in range(2):
        for j in range(3):
            axarr[i,j].imshow(G.forward(generate_random_seed(100), label_tensor)
              .detach().cpu().numpy().reshape(28,28), interpolation='none', cmap='Blues')
            pass
        pass
    pass
```

#### ✔ 훈련 반복문   
레이블 텐서를 판별기와 생성기에 전달해야 한다.   

```python
for epoch in range(epochs):
  print ("epoch = ", epoch + 1)

  for label, image_data_tensor, label_tensor in mnist_dataset:
    # 참에 대해 판별기 훈련
    D.train(image_data_tensor, label_tensor, torch.FloatTensor([1.0]))

    # 임의의 원핫 인코딩된 값을 레이블로 이용
    random_label = generate_random_one_hot(10)

    # 거짓에 대해 판별기 훈련
    # G의 기울기가 계산되지 않도록 detach() 함수를 이용
    D.train(G.forward(generate_random_seed(100), random_label).detach(), random_label, torch.FloatTensor([0.0]))
    
    # 임의의 원핫 인코딩된 값을 레이블로 이용
    # 판별기와 생성기 모두에 임의의 같은 레이블 텐서 투입
    random_label = generate_random_one_hot(10)

    # 생성기 훈련
    G.train(D, generate_random_seed(100), random_label, torch.FloatTensor([1.0]))

    pass
    
  pass
```

### **📊 조건부 GAN 결과 확인하기**   
#### 1️⃣ 판별기 손실값   
![image](https://user-images.githubusercontent.com/90624848/142964501-108d8d0f-9298-4651-a362-e3c24c594c89.png)   
👉 손실값은 완전히 0에 가깝지 않고 오히려 상승하는 추세로 보임   
👉 GAN의 이상적 손실값이 0이 아니므로 긍정적인 결과!

#### 2️⃣ 생성기 손실값   
![image](https://user-images.githubusercontent.com/90624848/142964515-fdc86d06-545c-45ce-9aca-59e7a9466b60.png)   
👉 GAN을 훈련할 때 추가적인 레이블 정보가 도움이 된다는 결론을 얻을 수 있음.   

#### 3️⃣ 숫자 5를 의미하는 이미지를 그리게하기   
![image](https://user-images.githubusercontent.com/90624848/142964529-b41ca714-4a0e-4d49-96c3-af7669d7819c.png)   
