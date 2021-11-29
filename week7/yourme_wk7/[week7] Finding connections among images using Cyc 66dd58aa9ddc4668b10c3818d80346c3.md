# [week7] Finding connections among images using CycleGAN(Yoorhim Cho)

**Cycle GAN** 

사진으로 구성되어있는 두 가지 데이터 셋이 있을 때, 한 데이터 셋의 사진들을 다른 데이터셋의 스타일로 바꾸어주는 것 + 동시에 반대쪽 수정작업도 동시에 훈련시킴.

**How does it work?**

1. pix2pix : pixel to pixel - 픽셀을 다른 픽셀로 바꾸어준다.
    1. **Supervised**: supervised learning framework
    2. loss: **Minimize the difference between output G(x) and ground truth y**
        
        ![Untitled](https://user-images.githubusercontent.com/90624848/143867143-3bc5a602-ce17-4be4-9ea7-53bd51d299bb.png)
        
    3. 정답에 가깝도록 중간값 선택 → 원하는 결과가 나오지 않음.
        
        극복 방안 ⇒ GAN
        
2. GAN 
    1.  x → G → G(x) → D → real or fake?
        1. D tries to identify the fakes
            
            ![Untitled 1](https://user-images.githubusercontent.com/90624848/143867198-e2facc95-93ac-4bc2-9357-cea0bbbe2298.png)
            
        2. G tries to synthesize fake images that fool D (realistic한 fake img 만들기)
            
            ![Untitled 2](https://user-images.githubusercontent.com/90624848/143867211-5fa8793f-9d4b-4e2d-971a-04a0f0e3a243.png)
            
    2. formulation
        
        ![Untitled 3](https://user-images.githubusercontent.com/90624848/143867214-e65be847-eb14-4acc-870f-c8eeafc5402d.png)
        
    3. G's perspective: D is a loss function.
        
        Rather than being hand-designed, it is learned.
        
        G and D grow together via competition
        
3. CycleGAN
    1. pix2pix: matching 페어가  같이 있음.
    2. Loss: L(G(x),y)
        
        G(x) should just look photorealistic
        
        **and be able to reconstruct x**
        
        **= and F(G(x)) should be F(G(x)) = x, where F is the inverse deep network**
        
    3. 문제: input을 무시하고 올바르지 않은 output 사진을 만들어 낼 수 있음.
    4. Loss formulation
        
        ![Untitled 4](https://user-images.githubusercontent.com/90624848/143867221-a4ba3356-8b0c-4652-8e27-caae0ca8b560.png)
        
    5. Training Details
        1. Generator G
            - Encoder-decoder
                - 이미지를 축약해서 핵심을 뽑고, 원하는 타겟 이미지 생성(SKT 디스코 GAN)
                
                ![Untitled 5](https://user-images.githubusercontent.com/90624848/143867226-e4dcb5b6-554b-4d60-84f1-3ae9c8c32bf8.png)
                
            - U-Net
                - Encoder-decoder에 skip connection 추가
                - 처음 디테일이 마지막 레이어까지 전달됨 → 디테일이 훨씬 더 많이 간직됨
                - 단점: 두 dataset이 비슷한 경우 skip connection을 많이 사용함
                    
                    → skip connection: depth가 거의 없어서 생성되는 결과가 만족스럽지 못함
                    
                
                ![Untitled 6](https://user-images.githubusercontent.com/90624848/143867237-916e6709-9621-43d0-9db3-9e1943425698.png)
                
            - ResNet
                - depth도 있고, detail을 간직할 수 있음.
                - 이미지 퀄리티 입장에서 가장 좋음
                - 단점: bottleneck이 없어서 메모리를 많이 사용함
                    
                    → 생성할 수 있는 learnable parameter가 적어짐
                    
                    → 많은 형태 변화를 만들어내지 못함
                    
                
                ![Untitled 7](https://user-images.githubusercontent.com/90624848/143867241-69f96047-5c80-44ef-864e-fa4e36021cdd.png)
                
        2. Objective
            - GANs with cross-entropy loss
                
                ![Untitled 8](https://user-images.githubusercontent.com/90624848/143867264-5a6e7a53-b496-415f-a41c-e6d7e31d379f.png)
                
                - loss training의 어려움: gradient가 flat해지는 현상이 발생함.
            - Least square GANs
                
                Stable training + better results
                
                ![Untitled 9](https://user-images.githubusercontent.com/90624848/143867271-e78cdda5-b419-4334-9e00-e6c1eb4c862d.png)
                
        3. replay buffer
            - 초기 Training: epoch를 거듭할수록 real이라 생각함
            - 다시 Training: 훈련할수록 fake라고 결론내림
                
                ⇒ Discriminator의 loss = Generator, Generator의 loss = Discriminator
                
                ⇒ 훈련이 Unstable
                
            - Solution
                - 1) Discriminator 여러개 생성 후 평균을 Generator에게 보여줌 ⇒ Generator의 수를 늘리는 것과 같음.
                    - 메모리 사용이 많아지게 됨.
                - 2) replay buffer
                    - 이전에 Generator가 만들어놨던 사진들을 Discriminator에게 주기적으로 보여줌.
                    - 예전 Generator가 어떻게 행동했는지 Discriminator가 함께 대응해야 해서 훨씬 안정적인 트레이닝이 가능함.
    6. Combine with as much L1 loss as possible
        1. L1 loss as a stable guiding force in GAN training(detail을 GAN에 의존)
            
            직접적인 L1은 구할 수 없지만, 유사한 L1을 사용
