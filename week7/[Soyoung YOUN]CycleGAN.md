# week7

# CycleGAN 동영상 내용 정리

## Reverse Prisma

: CycleGAN을 활용해 그림을 같은 composition과 내용을 가지고 있는 사진으로 바꿔줄 수 있는 인공지능을 개발함

## CycleGAN 역할

1. Reverse Prisma, 즉, 그림을 사진으로 바꿔주는 것 외에도 다른 여러 가지를 할 수 있음
2. 기본적으로, 사진으로 구성되어 있는 2가지 데이터셋이 있을 때, 한 데이터셋의 사진들을 다른 데이터셋의 이미지 스타일로 바꿔준다.
    
    Ex1)  여름에 찍은 사진들로 구성한 데이터셋 하나와 겨울에 찍은 사진들로 구성한 데이터셋 하나가 있을 때, 입력으로 여름 사진들을 넣고 훈련시키면 그 사진들을 겨울 사진와 같이 바꿔주기도 한다.
    
    Ex2) 얼굴을 라면으로 만들어줄 수도 있음!
    
3. 사진을 그림처럼 바꿔주는 Prisma 역할도 가능함

## PIX2PIX

= 픽셀을 다른 픽셀로 바꿔준다.

- input, ouput 모두 사진이어야 함
- matching pair가 존재함 → 훈련 시킬 때 이러한 pair 있으면 훈련이 용이하다.

### 예시 활용해 이해하기

Ex1) 흑백 사진을 컬러 사진으로 바뀌어준다.

- 사람이 정답을 붙여줄 필요가 없는 경우이다. → "Self-Supervised"
- test: 임의의 흑백 사진을 주고 이것이 어떤 컬러 사진으로 바뀌어서 나오는지 확인한다.
- train: 예시를 활용해보자면, 흑백 사진과 컬러 사진을 이용해 흑백 사진들이 어떠한 방식으로 패턴으로 컬러 사진으로 바뀌어야하는지 훈련한다.
- loss function: Minimize the difference between output G(x) (Deep Neural Network가 정의하는 함수임) and ground truth y

Ex2) 건물 앞부분 찍은 데이터셋들이 있고, 각각의 사진에 픽셀별로 벽, 창문, 문 등의 라벨을 한 데이터셋이 있는데, 라벨만 봤을 때 건물 앞부분 사진을 만들어낸다.

- Ex1과 달리 훈련과 테스트에 대해 정답을 부여할 필요가 있음 → "Supervised"

### Minimize the difference between output G(x) and ground truth y

1. 가장 쉽게 생각해볼 수 있는 것(L1 loss 라고 한다. → GAN과 함께 사용되어야 함)
    
    : y랑 G(x)를 픽셀 레벨에서 빼고, 그 차이를 최소화한다.
    

→ 차이를 최소화하는 개념은 현실에서 쉽지 않다! 흑백에서 컬러 사진으로 바꾸는 예제에서도 output은 ground truth보다 채도가 낮거나 흐릿한 정도로 나오는데, 이러한 과정에서 사람은 딱 눈으로 보았을 때 무언가 잘못되었음을 안다. 이렇게 사람이 알 수 있는 것을 인공지능도 알고 수행할 수 있어야 하는데, 이를 반영한 것이 바로 GAN이다.

⇒ L1 Loss + GAN(Reconstructed Loss)

### GAN

input x → G → G(x) → D → real or fake?

- D tries to identify the fakes
- G tries to synthesize fake images that fool D
    
    → arg maxD E_x, y[log D(G(x)) + log(1-D(y))]
    
    - 보통 D는 얼마나 잘 속였는지에 대한 확률(=input x가 fake일 확률)이다.
    - 가짜 이미지를 생성할 때, 이것이 높으면 높을수록 자신의 기능 잘 수행한 것이다.
    - 로그함수는 증가함수이므로 log D(G(x))를 max로 만드는 과정은 성능 높이는 것이다.
    
    → arg minG E_x, y[log D(G(x)) + log(1-D(y))]
    
    - 동시에 실제 사진(input y)을 보여주면, D는 낮으면 낮을수록(0에 가까울수록) 자신의 기능 잘 수행한 것이다.
    
    ⇒ arg minGmaxD E_x, y[log D(G(x)) + log(1-D(y))]  == L_GAN(G(x), y)
    
    - 이를 바탕으로 G와 D를 각각 알맞은 방향으로 훈련시킨다.
    - G tries to synthesize fake images that fool the best.

- G's perspective: D is a loss function.
- Rather than being hand-designed, it is learned.
- G and D grow together via competition

⇒  PIX2PIX는 픽셀 레벨에서의 이미지 차이와 바로 앞에서 말한 GAN Loss을 활용해서 Minimize를 진행한다.

## CycleGAN

: 언제나 matching pair가 존재할 수는 없다. 우리가 입력과 같은 composition은 아니지만, 그냥 사진은 여러 곳에서 얻을 수 있다. 이러한 상황에서 그럴 듯한 image transition을 하는 것이 cyclegan의 목표이다.

- L_GAN(G(x), y)는 계속해서 사용이 가능하다.
    
    → 문제가 발생함
    
    1. input을 무시하는 경우가 발생한다.
    2. 어떤 input이 들어오는지에 상관없이, 똑같은 output을 만들어내는 네트워크가 생성될 수 있다.

### CycleGAN's key objective

= G(x) should just look photorealistic and be able to reconstruct x(F(G(x)) should be F(G(x)) = x, where G is the inverse deep network)

### CycleGAN's loss formulation

<aside>
💡 L_GAN(G(x), y) + ||F(G(x)) - x||_1 + L_GAN(F(y), x) + ||G(F(y)) - y||_1

</aside>

Encoder-Decoder

- 디테일 보존이 안 됨(뭔가 압축/축약된 형식..?)
- Bottleneck 있음 → 여기서는 저장할 수 있는 것이 많지 않아 형태를 바꾸는 등의 급진적인 변화가 발생함
- 디테일 보존 어려움(고해상도 이미지 만들어내기 어려움)

U-NET

- 디테일이 그대로 전달되어 보존 가능함(skip connections)
- input, output이 컨텐츠가 비슷한 것에 활용이 가능함(Ex) 사진-사진 .... )

ResNET

- Depth도 있고 Bottleneck 없음 → 이미지 퀄리티 면에서 good!
- 메모리 굉장히 많이 사용함 → 우리가 생성할 수 있는 learnable parameters 수가 적어질 수 있다. → 많은 형태 변화 만들어내지 못함

### Training Details

1. GANS with cross-entropy loss
    - vanishing gradients 문제 발생한다.
    
    → 그래서 아래와 같은 LsGAN 사용한다.
    
2. Least square GANs
    - L_LSGAN(G, D_Y, X, Y) =  E_(y~p_data)(y)[(D_Y(y) - 1)^2] + E_(x~p_data)(y)[(D_Y(G(x))^2]
    - 1번은 확률에 로그를 씌움 → 여기서는 로그 사용하지 않음
    - 진짜인 경우 D가 1인 스코어 줘야 하고 가짜인 경우 0에 가까운 스코어 줘야 한다는 원리 사용했다.
    
    → No vanishing gradients ⇒ Stable training + better results (No noise, No mode collapse)
    
3. replay buffer
    - Discriminator의 loss = Generator, Generator의 loss = Discriminator 이기 때문이다.
    - because the discriminators can take very differenet trajectories in training
    
    → 이를 해결하기 위한 2가지 솔루션 존재함
    
    1. Discriminator 하나가 아니라 여러 개 생성하여 값 평균을 내어 Generator 한테 보여줌
        - 메모리 더 많이 사용한다는 단점 존재
    2. Reinforcement learning 에서 사용하는 replay buffer
        - 이 전에 Generator가 만들어놓았던 사진들을 주기적으로 다시 보여줌
            
            → 예전 Generator가 어떻게 행동했는지까지 Discriminator가 대응해야 하므로 훨씬 안정적이게 된다.
            
        - 즉, Discriminator만 학습시키는 것

### Combine with as much L1 Loss as possible

- L1 loss as a stable guiding force in GAN training

### Application on Domain Adaptation

1. 무인자동차
    
    CG2Real: GTA5 → real streetview,  real streetview → GTA5
    

### Q&A

Q1) 이미지 적용할 필요 없이 아키텍처 그대로 Machine translation 같은 곳에 사용할 수 있을 것 같은데, 어떻게 될 것 같은가?

A1) 이미지 적용 말고 다른 것들도 가능하다. 대표적인 예로 언어 분야에서도 사용할 수 있다.

Q2) Colorization이나 DeColorization 적용을 해보았는지?

A2) Colorization은 matching pair 만들기가 쉽기 때문에 cyclegan이 아니라 pix2pix에서 실험해봄.

Q3) 모네 그림에서 자연 풍경 사진으로 바꾸는 훈련 시, 자연 풍경이라는 것은 광범위한데, 이를 해결하기 위해 어떤 조건들을 걸었는가?

A3) 플리커 웹사이트에서 사진 얻음 → 태그를 landscape으로 해서 받음

Q4) 폭우, 폭설 등의 상황은 분석하기가 어려운데, 이런 상황에서도 위 기술들을 활용하면 그럴싸하게 나오는가? 

A4) 가능할 것이다. domain adaptation 연구 진행 시 비슷한 것 진행해보았는데, 꽤 잘 되었다고 한다..!!

Q5) 이러한 바꾸는 과정을 실시간으로 볼 수 있는건지? (아마 구현하고 사용자 배포 시를 묻는 것 같음..!)

A5) 해상도에 따라서 많이 달라지긴 하지만 500x500 픽셀 이미지를 만들어내는데 1초에 10장 정도의 이미지가 나왔는데, 좋은 gq 이용할 때임. 이를 실제로 구현해서 스마트폰에서 볼 수 있도록 하기 위해서는 다른 엔지니어링이 필요할 것 같다.