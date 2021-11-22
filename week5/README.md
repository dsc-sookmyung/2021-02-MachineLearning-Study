# 합성곱 GAN
* 이미지 데이터를 이용한 GAN 모델의 성능 향상을 위해 CNN 모델을 적용한다.
## 이미지 데이터의 특성
* 지역화 특성  
이미지는 feature 유무와 위치 정보를 학습에 사용하는 것이 모델 성능 향상에 도움이 된다.  
이는 CNN모델을 사용하면 가능하다.

* 합성곱 커널
특정 커널은 특정 feature를 추출하는 돋보기 기능을 한다. feature의 유무와 위치 정보는 이미지와 커널의 convolution 연산을 통해 알 수 있다. 연산 결과의 값이 클수록 해당 feature와 유사하다는 것을 알 수 있다.

## GAN에 CNN 적용하기
### 분류기
```python
self.model = nn.Sequential(
        nn.Conv2d(3,256,kernel_size=3,stride=2),
        nn.BatchNorm2d(256),
        nn.LeakyReLU(0.2),

        nn.Conv2d(256,256,kernel_size=3,stride=2),
        nn.BatchNorm2d(256),
        nn.LeakyReLU(0.2),

        nn.Conv2d(256,3,kernel_size=3,stride=2),
        nn.LeakyReLU(0.2),

        View(3*10),
        nn.Linear(3*10,1),
        nn.Sigmoid()
    )
```
* nn.Conv2d(3,256,kernel_size=3,stride=2) : 3 channel 이미지에서 256개의 feature 추출
* nn.Conv2d(256,256,kernel_size=3,stride=2) : 256 channel 이미지에서 256개 feature 추출
* nn.Conv2d(256,3,kernel_size=3,stride=2) : 256 channel 이미지에서 3개 feature 추출
* nn.Linear(3*10,1) : fully connected layer
### 생성기
```python
self.model = nn.Sequential(
        
        nn.Linear(54,3*13*23),
        nn.LeakyReLU(0.2),
        
        View((1,3,13,23)),

        nn.ConvTranspose2d(3,256,kernel_size=3,stride=2),
        nn.BatchNorm2d(256),
        nn.LeakyReLU(0.2),

        nn.ConvTranspose2d(256,256,kernel_size=3,stride=2),
        nn.BatchNorm2d(256),
        nn.LeakyReLU(0.2),

        nn.Conv2d(256,3,kernel_size=3,stride=2,padding=1),
        nn.BatchNorm2d(3),

        nn.Sigmoid()
    )
```
* nn.ConvTranspose2d(256,3,kernel_size=3,stride=2) : 3 channel 데이터 생성 
### 전치 합성곱 ConvTranspose2d
* 생성기는 큰 데이터를 작은 데이터로 압축시키는 분류기와 반대로 작은(seed) 데이터에서 큰 데이터를 생성한다.
* 예를 들어 3x3크기 입력 데이터가 nn.ConvTranspose2d(256,3,kernel_size=2,stride=1)를 통과하는 과정을 살펴본다.  
  입력층 : 3x3 크기 원본 데이터  
  중간 격자 : 원본데이터 각 픽셀 사이 사이에 0을 덧댄 7x7 크기 데이터  
  출력층 : 7x7 데이터와 2x2 필터를 convulotion 연산 후 6x6 크기 데이터 출력