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
  
![화면 캡처 2021-12-28 002546](https://user-images.githubusercontent.com/66189747/147485583-da251f09-3a7c-41b6-9543-1c10cc31e42f.png)
![화면 캡처 2021-12-28 002609](https://user-images.githubusercontent.com/66189747/147485589-1c492df8-7368-492b-9efc-84ed0ae5b485.png)
![화면 캡처 2021-12-28 002634](https://user-images.githubusercontent.com/66189747/147485592-5c28e1ad-7a12-4a82-88e1-7986fa9a081b.png)
![화면 캡처 2021-12-28 002648](https://user-images.githubusercontent.com/66189747/147485596-e4a944cc-9fa3-469b-ab64-46b7a3ec9fec.png)

