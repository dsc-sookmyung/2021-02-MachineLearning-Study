# Cycle GAN 
## 1. GAN
* GAN 구성  
   생성기(Generator) : 가짜 데이터 생성  
   분류기(Discriminator) : 진짜는 1 가짜는 0으로 분류  
* GAN 학습   
  $min_G$ $max_D$ $V(D,G) = E[ logD(x) ]+E[ log( 1-D(G(Z)) ) ]$  
  * Discriminator 학습 : $max_D$  
    $E[ logD(x) ]$ 는 $D(x) = 1$ 일 때 max  
    $E[ log( 1-D(G(Z)) ) ]$ 는 $D(G(x)) = 0$ 일 때 max  
    X는 진짜, Z->G(Z)는 가짜로 분류하며 학습
  * Generator 학습 : $min_G$  
    $E[ logD(x) ]$ 는 고정     
    $E[ log( 1-D(G(Z)) ) ]$ 는 $D(G(x)) = 1$ 일 때 min   
    X는 학습에 영향이 없고, Z->G(Z)를 D가 진짜로 분류하도록 학습

## 2. pix2pix
* pix2pix 구성    
   input : 흑백 이미지  
   output : 컬러 이미지
   model: 흑백 이미지 -> 컬러 이미지  
* pix2pix 학습   
  * self supervised learning  
    컬러이미지(원본,label)에서 흑백 추출(input data) 후 다시 컬러 복원(output data)  
  * loss_function  
    $Loss=\sum_{(x,y)} ||G(x)-Y||$ 
* 문제점  
  원본 데이터와 생성 데이터 차이큼 => GAN으로 보완
* 해결  
     $Loss=\sum_{(x,y)} ||G(x)-Y||+L_{GAN}(G(x),Y)$

## 3. Cycle GAN 
* Cycle GAN 구성  
  input : 그림 이미지 , 사진 이미지 (서로 연관 없음)  
  output : 사진 이미지  
  ex) {그림|새,말} {사진|호랑이, 강아지} => {사진|새,말}
* Cyle GAN 학습  
   * {$X$|그림} {사진} --G--> {$Y$|사진} --F-->{$\widehat{X}$|그림} --> {$X$|그림} {$\widehat{X}$|그림} 비교  
   * loss_function  
    $Loss=L_{GAN}(G(X),Y)+||F(G(X))-X||+L_{GAN}(F(Y),X)+||G(F(Y))-Y||$   
* Training Detail
   * ResNet 사용 : skip connection으로 디테일 전달 , bottle neck 없어서 메모리 사용
   * objective : loss function gradient vanishing 문제로 log에서 square로 변경
   * replay buffer : 여러 개의 Discriminator의 mean 값 사용