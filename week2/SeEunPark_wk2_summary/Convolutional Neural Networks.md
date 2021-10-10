# Convolutional Neural Networks

## Computer Vision

- 컴퓨터 비전은 다양한 분야에 응용되고 있음.
    - 주로 이미지 분류, 객체 인식, 신경망 스타일 번형 등
    - but, 입력데이터가 아주 크다는 것 → 합성곱 연산을 통해 해결 가능

## Edge detection Example

- 합성곱 작업 : 합성곱 신경망의 핵심 요소
    - 합성곱 연산
        - 원래 이미지 / 3*3 : 필터(커널) → 각각의 원소곱 후 전부 더해줌
            
            <img width="500" src="https://user-images.githubusercontent.com/66219968/136709858-f7ca244a-155a-41ee-bf63-16fd24eaa909.png">
            
        - 필터(커널)을 한칸 이동하여 합성곱 연산 진행 → 최종 4x4 새로운 행렬
            
            <img width="500" src="https://user-images.githubusercontent.com/66219968/136709862-67f96707-5ef8-4165-ba76-1b78b419cf88.png">
            
    - 수직 윤곽선 탐지법
        - 10과 0사이의 경계선이 수직 윤곽선
        - 필터를 통과해 합성곱 연산을 하게 되면 밝은 부분이 중앙으로 나타남. → 원래 이미지의 경계선
            
            <img width="500" src="https://user-images.githubusercontent.com/66219968/136709863-b71e58d2-f739-416c-b1d5-f496a4ab7052.png">
            
    

## More Edge Detection

- 양과 음의 윤곽선 차이 → 서로 다른 밝기의 전환
    
    <img width="500" src="https://user-images.githubusercontent.com/66219968/136709865-162f7710-99ce-4f9e-b1a4-793e1744f5a7.png">
    

## Padding

- 합성곱 방식의 단점 두 가지
    - 합성곱 연산을 거듭할수록 이미지는 축소됨
    - 가장자리 픽셀은 단 한 번만 사용됨 → 이미지 윤곽쪽의 정보를 버리게됨
    
- 합성곱 방식의 단점을 해결하기 위해 패딩 사용
    - 이미지 주위에 하나의 경계를 덧대는 것 → 이미지 크기가 커지므로 보통 숫자 0 사용
    - 최종 이미지 크기: (n + 2p - f + 1) x (n + 2p - f + 1)
        - n : 이미지 크기 / p : 패딩 크기 / f : 필터 크기
        - 일반적으로 필터의 크기는 홀수
            - 패딩이 비대칭 되기 때문. 홀수일 때 합성곱에서 동일한 크기로 패딩을 더해줄 수 있음.
            - 중심위치가 존재하기 때문
    

## Strided Convolutions

- 스트라이드 합성곱 : 합성곱 신경망의 기본 구성 요소
    - 스트라이드 : 필터의 이동횟수 → 스트라이드를 주게 되면 그 수만큼 필터가 이동해서 계산
- 최종 크기 : ((n+2p-f)/s+1) * ((n+2p-f)/s+1)

## Convolutions over volumes

- 이미지에 RGB가 들어가면 입체형으로변하게 되며, 차원이 하나 증가
    - 높이 x 넓이 x 채널로 변함 → 채널은 색상 또는 입체형의 이미지의 깊이
    - 합성곱에 사용되는 하나의 필터도 각 채널 별로 하나씩 증가
- 입체 이미지의 합성곱 계산 : 모든 채널의 합성곱 연산을 더해주는 형식
    - 각 채널 별로 필터는 모두 같을 수 있고, 다를 수 있음
- 패딩과 스트라이드가 없다고 가정했을 때, 최종 출력은
    
    **(n * n * n(c)) * ( f * f * n(c')) = (n - f + 1) * (n - f + 1) * n(c')**
    
    n: 이미지크기, n(c): 채널의 수, f: 필터의 크기, n(c'): 사용된 필터의 개수
    

## One Layer of a Convolutional Net

- 합성곱 신경망 한 계층의 구성 : 합성곱 연산 → 편향 추가 → 활성화 함수(비선형성을 적용하기 위함)
    
    <img width="500" src="https://user-images.githubusercontent.com/66219968/136709866-60ac5820-9204-4367-9d97-e66560108b6f.png">
    

## A Simple Convolution Network Example

<img width="500" src="https://user-images.githubusercontent.com/66219968/136709868-e17a6402-fae3-4046-b464-2fbe6c0fa6b0.png">

- 신경망 층의 구성
    - 합성곱 층
    - 풀링 층
    - 완전 연결 층
    

## Pooling Layers

- 풀링 층 사용 → 표현의 크기 줄임 → 계산 속도 줄임 → 특징 더 잘 검출 가능
- 최대 풀링, 평균 풀링. 주로 최대 풀링 사용.

<img width="500" src="https://user-images.githubusercontent.com/66219968/136709870-47e8f7f0-76dc-4030-af81-f8da620b75d5.png">
## CNN Example

- LeNet-5 라는 사용한 고전적인 신경망과 유사한 구조
    
    <img width="500" src="https://user-images.githubusercontent.com/66219968/136709871-69b25167-db4d-485c-a882-e9c2c0f7fe61.png">
    
- 합성곱 신경망의 분야에는 두 종류의 관습
    - 합성곱 층과 풀링 층을 하나의 층으로 간주 → 사용!
    - 합성곱 층과 풀링 층을 각각의 층으로 간주
        
        <img width="500" src="https://user-images.githubusercontent.com/66219968/136709872-4995f2f0-e8a2-433b-8491-3ca54d0ab992.png">
        
        - Point
            - 최대 풀링 층은 변수가 따로 없음
            - 합성곱 층이 상대적으로 적은 변수를 가짐 (신경망의 대부분의 변수는 완전 연결 층에 있음)
            - 활성값의 크기도 신경망이 깊어질수록 점점 감소

## Why Convolutions

- 변수 공유
    - 어떤 한 부분에서 이미지 특성 검출하는 필터가 이미지의 다른 부분에서도 똑같이 적용되거나 도움됨
- 희소 연결
    - 출력값이 이미지의 일부의 영향을 받음. 나머지 픽셀들은 영향을 받지 않음. → 과대 적합 방지 가능
- 이동불변성을 포착하는데 용이
    - 이미지가 약간의 변형이 있어도 포착 가능
