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
