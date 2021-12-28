### ❄ CycleGAN Reference   

👩‍💻 [Lecture](https://youtu.be/4LktBHGCNfw)   
💻 [Code Reference Github](https://github.com/aladdinpersson/Machine-Learning-Collection/tree/master/ML/Pytorch/GANs/CycleGAN)  

💾 [Dataset](https://www.kaggle.com/suyashdamle/cyclegan?select=horse2zebra)   
🌟 [Official Code](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)   
⭐ [Simplified Version](https://github.com/aitorzip/PyTorch-CycleGAN)

‼ I didn't upload the dataset & etc. files bcuz...   
  　**TOO LARGE** to upload to GITHUB XD   
# Cycle GAN
## 참고한 링크
https://www.tensorflow.org/tutorials/generative/cyclegan  
## 생성기와 판별자 가져오기를 지원하는 패키지 설치, import
![패키지 설치 및 import](https://user-images.githubusercontent.com/66189747/147479955-10f1c607-de65-4c1d-ae25-6148d3eef883.png)
## 입력파이프라인 생성
이미지를 가지고 온 후, 이를 랜덤하게 이미지를 수정한다  
  
무작위 지터링 : 이미지 286x286크기로 조정후 256x256으로 무작위로 자름  
무작위 미러링 : 이미지 좌우로 무작위로 뒤집힘  
  
![이미지 가져온다](https://user-images.githubusercontent.com/66189747/147480738-0cf8198f-d204-42dd-83c1-cea6dd9a5972.png)   
이미지 가져온 후 BUFFER_SIZE와 BATCH_SIZE, IMG_WIDTH, IMG_HIGHT를 설정한다  
  
![변수 설정](https://user-images.githubusercontent.com/66189747/147481022-2b699e1a-ef58-4083-9dce-820c3fa1e970.png)  
  
함수 만들고 이미지들 랜덤하게 수정한다  
  
![함수 선언](https://user-images.githubusercontent.com/66189747/147481188-300e6b7b-a2d5-4b5c-b924-8507eb2c3834.png)
![함수 선언 및 이미지 수정](https://user-images.githubusercontent.com/66189747/147481196-aee4b43a-40f8-4219-b9ed-b9ce92cc9596.png)
  
수정한 이미지 확인하기
  
![수정한 말 이미지](https://user-images.githubusercontent.com/66189747/147481281-ebb1c697-853e-4a36-9ae8-49f6c7fa39c5.png)
![수정한 얼룩말 이미지](https://user-images.githubusercontent.com/66189747/147481292-69689a02-5216-49a5-84e5-9cd78f4d78cc.png)

## Pix2Pix 모델 가져오기
설치된 tensorflow_examples 패키지를 통해 Pix2Pix에서 사용되는 생성기와 판별자를 가져온다.  
이 튜토리얼에서 사용된 모델 아키텍처는 Pix2Pix에서 사용된 것과 매우 유사하다.  
### Pix2Pix와 CycleGan의 차이점  
<ol><li>CycleGan은 배치정규화 대신 인스턴스 정규화를 사용한다</li>
<li>논문에서는 수정된 resnet기반 생성기 이용, 여기서는 단순화를 위해 수정된 unet 생성기를 이용한다</li>  
### 2개의 생성기 및 2개의 판별자 훈련  
생성기 G는 이미지 X를 Y로 변환하는 방법 학습  
생성기 F는 이미지 Y를 X로 변환하는 방법 학습  
판별자 D_X는 이미지 X와 생성된 이미지 X를 구별하는 방법 학습  
판별자 D_Y는 이미지 Y와 생성된 이미지 Y를 구별하는 방법 학습  
  
![Pix2Pix의 generator_g, f, discriminator_x, y가져오기](https://user-images.githubusercontent.com/66189747/147485583-da251f09-3a7c-41b6-9543-1c10cc31e42f.png)
![generator, discriminator훈련](https://user-images.githubusercontent.com/66189747/147485589-1c492df8-7368-492b-9efc-84ed0ae5b485.png)
![generator, discriminator훈련 결과](https://user-images.githubusercontent.com/66189747/147485592-5c28e1ad-7a12-4a82-88e1-7986fa9a081b.png)
![generator, discriminator훈련, 결과](https://user-images.githubusercontent.com/66189747/147485596-e4a944cc-9fa3-469b-ab64-46b7a3ec9fec.png)
  
## 손실함수
CycleGan에는 훈련할 쌍으로 연결된 데이터가 없어 훈련 중에 입력 x와 대상 y의 쌍이 언제나 의미가 있다고 할 수 없다  
네트워크가 올바른 매핑을 학습하도록 강제하기 위해 **주기일관성손실**을 제안    
판별자 손실 및 생성기 손실은 pix2pix에서 사용된 것과 유사하다  
  
### 주기일관성  
결과가 원래 입력에 가까워야 함을 의미한다  
ex)문장을 영어에서 프랑스어로 번역한 다음 다시 프랑스어에서 영어로 번역하면 결과 문장과 원래 문장이 같아야한다  
X-G생성기->Y->F생성기->X  
처음 x와 마지막 X사이에 평균절대오차가 계산된다  
![loss 관련 함수 선언](https://user-images.githubusercontent.com/66189747/147533008-0de89806-48f4-4637-99dc-26125c5aa357.png)  
![loss 관련 함수 선언 및 생성기와 판별자의 옵티마이저 초기화, 체크포인트 생성](https://user-images.githubusercontent.com/66189747/147533014-cda9df75-0e1b-45c5-9313-12b9b6e7e032.png)  
  
## 훈련하기
이번 예제에서는 논문과 달리 훈련 시간을 줄이기 위해 40epoch를 대상으로 훈련한다  
-> 그렇기 때문에 예측 정확성 떨어질 수 있다  
![EPOCHS선언 및 이미지 생성 함수 선언](https://user-images.githubusercontent.com/66189747/147533217-13ea17b5-ae70-444d-bc11-7686a46dcf42.png)  
훈련루프 네가지 기본 단계
<ol><li>예측</li>
<li>손실을 계산</li>
<li>역전파를 사용하여 그래디언트를 계산</li>
<li>그래디언트를 옵티마이저에 적용</li></ol>  
![생성자 G와 X훈련 함수 생성](https://user-images.githubusercontent.com/66189747/147533347-8ee6c446-c624-4dce-b8a5-3ea2f770d0dd.png)
![생성자 G와 X훈련 함수 생성](https://user-images.githubusercontent.com/66189747/147533350-8d2543bd-8678-4151-a253-e5770d245350.png)
![훈련](https://user-images.githubusercontent.com/66189747/147533422-12c95bb5-18c0-48c9-a33a-deff927e9ad5.png)
![실행화면 / 런타임 관련 문제로 훈련을 완료하지 못함](https://user-images.githubusercontent.com/66189747/147533466-a092881c-8da6-431d-8eba-639eaa0d8e7b.png)
![이미지 생성하기(최종)](https://user-images.githubusercontent.com/66189747/147533542-c039d91e-f933-4b34-b2b0-48dd9a5951b0.png)

