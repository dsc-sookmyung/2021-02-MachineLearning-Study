### โ CycleGAN Reference   

๐ฉโ๐ป [Lecture](https://youtu.be/4LktBHGCNfw)   
๐ป [Code Reference Github](https://github.com/aladdinpersson/Machine-Learning-Collection/tree/master/ML/Pytorch/GANs/CycleGAN)  

๐พ [Dataset](https://www.kaggle.com/suyashdamle/cyclegan?select=horse2zebra)   
๐ [Official Code](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)   
โญ [Simplified Version](https://github.com/aitorzip/PyTorch-CycleGAN)

โผ I didn't upload the dataset & etc. files bcuz...   
  ใ**TOO LARGE** to upload to GITHUB XD   
# Cycle GAN
## ์ฐธ๊ณ ํ ๋งํฌ
https://www.tensorflow.org/tutorials/generative/cyclegan  
## ์์ฑ๊ธฐ์ ํ๋ณ์ ๊ฐ์ ธ์ค๊ธฐ๋ฅผ ์ง์ํ๋ ํจํค์ง ์ค์น, import
![ํจํค์ง ์ค์น ๋ฐ import](https://user-images.githubusercontent.com/66189747/147479955-10f1c607-de65-4c1d-ae25-6148d3eef883.png)
## ์๋ ฅํ์ดํ๋ผ์ธ ์์ฑ
์ด๋ฏธ์ง๋ฅผ ๊ฐ์ง๊ณ  ์จ ํ, ์ด๋ฅผ ๋๋คํ๊ฒ ์ด๋ฏธ์ง๋ฅผ ์์ ํ๋ค  
  
๋ฌด์์ ์งํฐ๋ง : ์ด๋ฏธ์ง 286x286ํฌ๊ธฐ๋ก ์กฐ์ ํ 256x256์ผ๋ก ๋ฌด์์๋ก ์๋ฆ  
๋ฌด์์ ๋ฏธ๋ฌ๋ง : ์ด๋ฏธ์ง ์ข์ฐ๋ก ๋ฌด์์๋ก ๋ค์งํ  
  
![์ด๋ฏธ์ง ๊ฐ์ ธ์จ๋ค](https://user-images.githubusercontent.com/66189747/147480738-0cf8198f-d204-42dd-83c1-cea6dd9a5972.png)   
์ด๋ฏธ์ง ๊ฐ์ ธ์จ ํ BUFFER_SIZE์ BATCH_SIZE, IMG_WIDTH, IMG_HIGHT๋ฅผ ์ค์ ํ๋ค  
  
![๋ณ์ ์ค์ ](https://user-images.githubusercontent.com/66189747/147481022-2b699e1a-ef58-4083-9dce-820c3fa1e970.png)  
  
ํจ์ ๋ง๋ค๊ณ  ์ด๋ฏธ์ง๋ค ๋๋คํ๊ฒ ์์ ํ๋ค  
  
![ํจ์ ์ ์ธ](https://user-images.githubusercontent.com/66189747/147481188-300e6b7b-a2d5-4b5c-b924-8507eb2c3834.png)
![ํจ์ ์ ์ธ ๋ฐ ์ด๋ฏธ์ง ์์ ](https://user-images.githubusercontent.com/66189747/147481196-aee4b43a-40f8-4219-b9ed-b9ce92cc9596.png)
  
์์ ํ ์ด๋ฏธ์ง ํ์ธํ๊ธฐ
  
![์์ ํ ๋ง ์ด๋ฏธ์ง](https://user-images.githubusercontent.com/66189747/147481281-ebb1c697-853e-4a36-9ae8-49f6c7fa39c5.png)
![์์ ํ ์ผ๋ฃฉ๋ง ์ด๋ฏธ์ง](https://user-images.githubusercontent.com/66189747/147481292-69689a02-5216-49a5-84e5-9cd78f4d78cc.png)

## Pix2Pix ๋ชจ๋ธ ๊ฐ์ ธ์ค๊ธฐ
์ค์น๋ tensorflow_examples ํจํค์ง๋ฅผ ํตํด Pix2Pix์์ ์ฌ์ฉ๋๋ ์์ฑ๊ธฐ์ ํ๋ณ์๋ฅผ ๊ฐ์ ธ์จ๋ค.  
์ด ํํ ๋ฆฌ์ผ์์ ์ฌ์ฉ๋ ๋ชจ๋ธ ์ํคํ์ฒ๋ Pix2Pix์์ ์ฌ์ฉ๋ ๊ฒ๊ณผ ๋งค์ฐ ์ ์ฌํ๋ค.  
### Pix2Pix์ CycleGan์ ์ฐจ์ด์   
<ol><li>CycleGan์ ๋ฐฐ์น์ ๊ทํ ๋์  ์ธ์คํด์ค ์ ๊ทํ๋ฅผ ์ฌ์ฉํ๋ค</li>
<li>๋ผ๋ฌธ์์๋ ์์ ๋ resnet๊ธฐ๋ฐ ์์ฑ๊ธฐ ์ด์ฉ, ์ฌ๊ธฐ์๋ ๋จ์ํ๋ฅผ ์ํด ์์ ๋ unet ์์ฑ๊ธฐ๋ฅผ ์ด์ฉํ๋ค</li>  
### 2๊ฐ์ ์์ฑ๊ธฐ ๋ฐ 2๊ฐ์ ํ๋ณ์ ํ๋ จ  
์์ฑ๊ธฐ G๋ ์ด๋ฏธ์ง X๋ฅผ Y๋ก ๋ณํํ๋ ๋ฐฉ๋ฒ ํ์ต  
์์ฑ๊ธฐ F๋ ์ด๋ฏธ์ง Y๋ฅผ X๋ก ๋ณํํ๋ ๋ฐฉ๋ฒ ํ์ต  
ํ๋ณ์ D_X๋ ์ด๋ฏธ์ง X์ ์์ฑ๋ ์ด๋ฏธ์ง X๋ฅผ ๊ตฌ๋ณํ๋ ๋ฐฉ๋ฒ ํ์ต  
ํ๋ณ์ D_Y๋ ์ด๋ฏธ์ง Y์ ์์ฑ๋ ์ด๋ฏธ์ง Y๋ฅผ ๊ตฌ๋ณํ๋ ๋ฐฉ๋ฒ ํ์ต  
  
![Pix2Pix์ generator_g, f, discriminator_x, y๊ฐ์ ธ์ค๊ธฐ](https://user-images.githubusercontent.com/66189747/147485583-da251f09-3a7c-41b6-9543-1c10cc31e42f.png)
![generator, discriminatorํ๋ จ](https://user-images.githubusercontent.com/66189747/147485589-1c492df8-7368-492b-9efc-84ed0ae5b485.png)
![generator, discriminatorํ๋ จ ๊ฒฐ๊ณผ](https://user-images.githubusercontent.com/66189747/147485592-5c28e1ad-7a12-4a82-88e1-7986fa9a081b.png)
![generator, discriminatorํ๋ จ, ๊ฒฐ๊ณผ](https://user-images.githubusercontent.com/66189747/147485596-e4a944cc-9fa3-469b-ab64-46b7a3ec9fec.png)
  
## ์์คํจ์
CycleGan์๋ ํ๋ จํ  ์์ผ๋ก ์ฐ๊ฒฐ๋ ๋ฐ์ดํฐ๊ฐ ์์ด ํ๋ จ ์ค์ ์๋ ฅ x์ ๋์ y์ ์์ด ์ธ์ ๋ ์๋ฏธ๊ฐ ์๋ค๊ณ  ํ  ์ ์๋ค  
๋คํธ์ํฌ๊ฐ ์ฌ๋ฐ๋ฅธ ๋งคํ์ ํ์ตํ๋๋ก ๊ฐ์ ํ๊ธฐ ์ํด **์ฃผ๊ธฐ์ผ๊ด์ฑ์์ค**์ ์ ์    
ํ๋ณ์ ์์ค ๋ฐ ์์ฑ๊ธฐ ์์ค์ pix2pix์์ ์ฌ์ฉ๋ ๊ฒ๊ณผ ์ ์ฌํ๋ค  
  
### ์ฃผ๊ธฐ์ผ๊ด์ฑ  
๊ฒฐ๊ณผ๊ฐ ์๋ ์๋ ฅ์ ๊ฐ๊น์์ผ ํจ์ ์๋ฏธํ๋ค  
ex)๋ฌธ์ฅ์ ์์ด์์ ํ๋์ค์ด๋ก ๋ฒ์ญํ ๋ค์ ๋ค์ ํ๋์ค์ด์์ ์์ด๋ก ๋ฒ์ญํ๋ฉด ๊ฒฐ๊ณผ ๋ฌธ์ฅ๊ณผ ์๋ ๋ฌธ์ฅ์ด ๊ฐ์์ผํ๋ค  
X-G์์ฑ๊ธฐ->Y->F์์ฑ๊ธฐ->X  
์ฒ์ x์ ๋ง์ง๋ง X์ฌ์ด์ ํ๊ท ์ ๋์ค์ฐจ๊ฐ ๊ณ์ฐ๋๋ค  
![loss ๊ด๋ จ ํจ์ ์ ์ธ](https://user-images.githubusercontent.com/66189747/147533008-0de89806-48f4-4637-99dc-26125c5aa357.png)  
![loss ๊ด๋ จ ํจ์ ์ ์ธ ๋ฐ ์์ฑ๊ธฐ์ ํ๋ณ์์ ์ตํฐ๋ง์ด์  ์ด๊ธฐํ, ์ฒดํฌํฌ์ธํธ ์์ฑ](https://user-images.githubusercontent.com/66189747/147533014-cda9df75-0e1b-45c5-9313-12b9b6e7e032.png)  
  
## ํ๋ จํ๊ธฐ
์ด๋ฒ ์์ ์์๋ ๋ผ๋ฌธ๊ณผ ๋ฌ๋ฆฌ ํ๋ จ ์๊ฐ์ ์ค์ด๊ธฐ ์ํด 40epoch๋ฅผ ๋์์ผ๋ก ํ๋ จํ๋ค  
-> ๊ทธ๋ ๊ธฐ ๋๋ฌธ์ ์์ธก ์ ํ์ฑ ๋จ์ด์ง ์ ์๋ค  
![EPOCHS์ ์ธ ๋ฐ ์ด๋ฏธ์ง ์์ฑ ํจ์ ์ ์ธ](https://user-images.githubusercontent.com/66189747/147533217-13ea17b5-ae70-444d-bc11-7686a46dcf42.png)  
ํ๋ จ๋ฃจํ ๋ค๊ฐ์ง ๊ธฐ๋ณธ ๋จ๊ณ
<ol><li>์์ธก</li>
<li>์์ค์ ๊ณ์ฐ</li>
<li>์ญ์ ํ๋ฅผ ์ฌ์ฉํ์ฌ ๊ทธ๋๋์ธํธ๋ฅผ ๊ณ์ฐ</li>
<li>๊ทธ๋๋์ธํธ๋ฅผ ์ตํฐ๋ง์ด์ ์ ์ ์ฉ</li></ol>  
    
![์์ฑ์ G์ Xํ๋ จ ํจ์ ์์ฑ](https://user-images.githubusercontent.com/66189747/147533347-8ee6c446-c624-4dce-b8a5-3ea2f770d0dd.png)  
![์์ฑ์ G์ Xํ๋ จ ํจ์ ์์ฑ](https://user-images.githubusercontent.com/66189747/147533350-8d2543bd-8678-4151-a253-e5770d245350.png)  
![ํ๋ จ](https://user-images.githubusercontent.com/66189747/147533422-12c95bb5-18c0-48c9-a33a-deff927e9ad5.png)  
![์คํํ๋ฉด / ๋ฐํ์ ๊ด๋ จ ๋ฌธ์ ๋ก ํ๋ จ์ ์๋ฃํ์ง ๋ชปํจ](https://user-images.githubusercontent.com/66189747/147533466-a092881c-8da6-431d-8eba-639eaa0d8e7b.png)  
![์ด๋ฏธ์ง ์์ฑํ๊ธฐ(์ต์ข)](https://user-images.githubusercontent.com/66189747/147533542-c039d91e-f933-4b34-b2b0-48dd9a5951b0.png)  

