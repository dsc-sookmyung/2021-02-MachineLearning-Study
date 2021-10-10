# 1. Convolutional Neural Networks

### Computer Vision

- 자율주행, 얼굴인식, 예술 등 다양한 분야에 응용되고 있음.
- 컴퓨터비전의 발달 : 빠른 발전이 많은 새로운 application들이 만들어지게 함
- 주로 다루는 문제: 이미지분류, 객체인식, 신경망 스타일 변형 등
- Problems : 입력 데이터가 아주 크다. (ex. 이미지의 크기가 계속 커지는 것)
    
    ⇒  합성곱 연산을 통해 해결할 수 있음.
    

### Edge Detection Examples

- 이미지 = (높이) $\times$ (넓이)
- 합성곱 연산
    
    ![Untitled](1%20Convolutional%20Neural%20Networks%20b009a0f44e8c4c09bda778b8a4bbb0f2/Untitled.png)
    
    - 6*6 Original Image와 3*3 Kernel(Filter) 행렬을 합성곱
    
    ![Untitled](1%20Convolutional%20Neural%20Networks%20b009a0f44e8c4c09bda778b8a4bbb0f2/Untitled%201.png)
    
    - step으로 커널을 한 칸 이동하여 합성곱 연산을 진행하여 최종 4*4의 새로운 행렬을 만들어냄.
    
    ![Untitled](1%20Convolutional%20Neural%20Networks%20b009a0f44e8c4c09bda778b8a4bbb0f2/Untitled%202.png)
    
    - 수직 윤곽선 탐지 필터
    - 커널을 통과하여 합성곱 연산을 하게 되면 밝은 부분이 중앙으로 나타남. ⇒ 원래 이미지의 경계선에 해당
    - 경계선이 두꺼운 이유 ⇒ 이미지가 작아서
    
    ### More Edge Detection
    
    ![Untitled](1%20Convolutional%20Neural%20Networks%20b009a0f44e8c4c09bda778b8a4bbb0f2/Untitled%203.png)
    
- Vertical : 세로 윤곽선 검출
- Horizontal : 가로 윤곽선 검출
- Sobel : 중간 부분의 픽셀에 더 중점을 두는 것 ⇒ 더 선명해 보임
- Scharr
- 어떤 복잡한 이미지에서 윤곽선을 검출하려고 할 때
    - 스스로 학습하게 두고 9개의 숫자를 아예 변수로 설정
        
        ⇒ 신경망이 윤곽선같은 하위 단계의 속성을 학습할 수 있게 됨.
        
    - 역전달(역전파) 형식으로 학습된 9개의 숫자가 문제에 적합한 필터를 만드는 방법을 사용

### Padding

- 이전에 사용했던 방법의 두 가지 단점
    1. 계속 합성곱 연산 ⇒ 이미지가 계속 축소
        
        ⇒ 수백 개의 층에서 각 층마다 축소된다면 모든 층을 거친 뒤에는 아주 작은 이미지만 남음
        
    2. 가장자리 픽셀은 단 한 번만 연산에 참여함. ⇒ 이미지 윤곽쪽의 정보를 버리게 됨
    
    ⇒ 해결 : Padding
    
- 이미지 주위에 추가로 하나의 경계를 덧대는 것(보통 0을 사용) ⇒ 이미지 크기가 조금 커짐
- 최종 이미지 크기 = (n + 2p - f + 1) × (n + 2p -f + 1)
    - n : 이미지 크기
    - p : 패딩 크기
    - f : 필터 크기
        - 일반적으로 필터의 크기는 홀수
            1. 패딩이 비대칭이 됨 ⇒ 합성곱에서 동일한 크기로 패딩을 더해줄 수 있음
            2. 중심위치가 존재함.
- 패딩을 얼마만큼 할 것인가?
    1. Valid Convolution : **No padding**
        - (n×n) * (f×f) → (n - f + 1) × (n - f + 1)
    2. Same Convolution : Pad so that ouput size is the same as the input size
        - (n + 2p - f + 1) × (n + 2p - f + 1)
            
            n + 2p - f + 1 = n
            
            p = (f - 1) / 2
            
    

### Strided Convolutions

- 합성곱 신경망의 기본 구성 요소
- Stride : 필터의 이동 횟수
- 이미지의 최종 크기
    
    ![Untitled](1%20Convolutional%20Neural%20Networks%20b009a0f44e8c4c09bda778b8a4bbb0f2/Untitled%204.png)
    
    - (n+2p-f)/s가 정수가 아닌 경우 ⇒ 소수점 내림
    - 보통 필터에 맞춰서 최대한 정수가 될 수 있도록 패딩과 스트라이드 설정
- 신호처리에서의 교차상관과 합성곱의 관계
    - 합성곱 : 필터를 가로축과 세로축으로 뒤집는 연산을 해야함 → 교차상관 : 미러링 없이 연산하는 것
    - 딥러닝에서는 심층 신경망 분야에서는 영향이 없으므로 생략

### Convolutions Over Volumes

- 이미지에 색상(RGB)이 들어가면 입체형으로 변하게 됨
    
    → 차원의 증가 : (height) × (width) × (#channels)
    
    - (이미지의 채널 수) = (필터의 채널 수)
- Filter : 3D 필터 사용 → 채널 별로 하나씩 증가
- Convolutions on RGB image
    
    ![Untitled](1%20Convolutional%20Neural%20Networks%20b009a0f44e8c4c09bda778b8a4bbb0f2/Untitled%205.png)
    
    - 패딩과 스트라이드가 없다고 가정했을때의 최종 출력:
        
        ![Untitled](1%20Convolutional%20Neural%20Networks%20b009a0f44e8c4c09bda778b8a4bbb0f2/Untitled%206.png)
        
    - 여러 필터를 사용할 때 : 첫 번째를 앞 쪽에 두고 두 번째 필터 결과를 뒤에 놓음.
        
        ![Untitled](1%20Convolutional%20Neural%20Networks%20b009a0f44e8c4c09bda778b8a4bbb0f2/Untitled%207.png)
        

### One Layer of a Convolutional Net

- 합성곱 연산 → 편향 추가 → 활성화 함수
    - 활성화 함수 : 비선형성을 적용하기 위함 (보통 ReLU를 많이 사용)
- 이미지의 크기가 어떻든 변수의 수는 고정되어 있음
    - 3×3×3 + 1 → 28 * 10(#filter) = 280 (고정)
    - n개의 필터로 여러가지 다른 속성들을 검출할 수 있음
    - 아주 큰 이미지라도 적은 수의 변수로 가능함
    
    ⇒ 과대적합을 방지하는 합성곱 신경망의 성질
    
- 표기법
    
    ![Untitled](1%20Convolutional%20Neural%20Networks%20b009a0f44e8c4c09bda778b8a4bbb0f2/Untitled%208.png)
    
- $l$번째 층의 연산
    - Input : 이전 층 $(l-1)$의 이미지 크기
        
        ![Untitled](1%20Convolutional%20Neural%20Networks%20b009a0f44e8c4c09bda778b8a4bbb0f2/Untitled%209.png)
        
    - Output : 결과로 나오는 이미지 크기
        
        ![Untitled](1%20Convolutional%20Neural%20Networks%20b009a0f44e8c4c09bda778b8a4bbb0f2/Untitled%2010.png)
        
    - $l$번째 층의 높이 및 넓이의 크기연산 공식
        
        ![Untitled](1%20Convolutional%20Neural%20Networks%20b009a0f44e8c4c09bda778b8a4bbb0f2/Untitled%2011.png)
        

### Simple Convolutional Network Example

![Untitled](1%20Convolutional%20Neural%20Networks%20b009a0f44e8c4c09bda778b8a4bbb0f2/Untitled%2012.png)

### Pooling Layers

- Max and Average Pooling
    
    ![Untitled](1%20Convolutional%20Neural%20Networks%20b009a0f44e8c4c09bda778b8a4bbb0f2/Untitled%2013.png)
    
    - Max Pooling (최대 풀링)
        - 이미지의 특징이 필터의 한 부분에서 검출되면 큰 수를 남김.
        - 그렇지 않으면 다른 최대값들에 비해 상대적으로 작아져서 특징을 더 잘 남긴다.
    - Average Pooling (평균 풀링)
        - output : 필터 크기에 포함된 모든 원소의 평균값
        - 특징을 포착해야하는 합성곱의 특성상 Max Pooling을 더 많이 씀

### CNN Example

- CNN 예시
    
    ![Untitled](1%20Convolutional%20Neural%20Networks%20b009a0f44e8c4c09bda778b8a4bbb0f2/Untitled%2014.png)
    
    - 두 종류의 관습
        1. 합성곱과 풀링층을 하나의 층으로 보는 것
        2. 합성곱 층과 풀링층을 각각의 층으로 간주하는 것
        - LeNet - 5에서는 Pooling 층에 학습해야 할 변수가 없어 1번으로 선택
    - 보통 Conv1 → Pool1  →  Conv2 → Pool2 → Flatten → FC → FC → Softmax 순서
    - Max Pooling 층은 변수를 갖지 않음
    - Convolutional Layer에서는 변수를 적게 사용함
    - 합성곱 신경망이 깊어질수록 크기는 작아짐

### Why Convolutions

- 합성곱 신경망을 사용하면 변수를 적게 사용할  수 있음.
    1. 변수 공유 ⇒ 필터가 이미지의 다른 부분에서도 똑같이 적용 or 도움이 됨.
    2. 희소 연결 ⇒ 출력값이 이미지의 일부에 영향을 받고, 나머지 픽셀들의 영향을 받지 않음 ⇒ 과대적합 방지
- 이동불변성 포착에 용이함 ⇒ 약간의 변형이 이미지 상에 있어도 포착 가능