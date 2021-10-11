# Week2 ML study
신경망 - 층, 은닉유닛 수, 학습률, 활성화함수 등을 정해야 하는데 곧바로 가장 적절한 값을 찾을 수 없기 때문에 반복적인 과정을 거쳐야 한다
아이디어 -> 특정개수의 층과 유닛, 특정 데이터 세트에 맞는 신경망을 만든다 -> 코딩, 실행, 실험 등 진행 -> 아이디어 개선 (반복)

## 신경망 세트
trainning set : 훈련 알고리즘 제작
dev set : 교차검증 개발세트, 서로 다른 알고리즘 중 가장 좋은 성능을 내는 것을 찾는다
test set : 최종 모델 알고리즘이 얼마나 잘 작동하는지 test
과거 : trainning set 70% | test set 30% 또는
       trainning set 60% | dev set 20% | test set 20%
       => 데이터의 수가 많아짐에 따라 dev와 test 셋의 수가 20%나 차지할 필요가 없어짐
현재 : trainning set 98% | dev set 1% | test set 1% 또는
       trainning set 99.5% | dev set 0.5% | test set 0.5%
       
       최종 네트워크의 비편향 추정이 필요한 경우에만 테스트 세트가 필요
       -> 테스트 세트를 사용하지 않는 경우 존재
          테스트 세트라고 하지만 실제로는 교차검증을 위해 사용하는 경우 (test세트가 아니라 dev세트인 경우)
       
## 편향과 분산
train set Error와 dev set Error을 비교함으로 편향문제와 분산문제를 파악할 수 있다
##### train set error : 훈련데이터에서 알고리즘이 얼마나 적합한지 감을 잡을 수 있다, 편향문제
##### train set error와 dev set error와의 차이 : 분산문제가 얼마나 나쁜지 감을 잡을 수 있다, 일반화를 잘 하느냐에 따라 분산에 대한 감이 달라진다
높은 편향값 : 데이터 과소적합
높은 분산 : 데이터 과대적합
ex) 인간의 오차(최적 오차, 베이즈 오차) ~ 0인경우
    train set error : 1% dev set error : 11% => 데이터 과대적합, 일반화가 제대로 되지 않음 -> 높은 분산을 가짐
    train set error : 15% dev set error : 16% => 상대적으로 훈련데이터에 잘 맞지 않지만(데이터 과소적합) train set error와 dev set error 간 차이가 1%밖에 되지 않기 때문에 합리적 수준의 개발세트로 사용된다
    train set error : 15% dev set error : 30% => 데이터 과소적합, 일반화가 제대로 되지 않음 -> 높은 편향과 분산을 가짐
    train set error : 0.5% dev set error : 1% => 낮은 편향과 낮은 분산을 가짐
    
    인간의 오차(베이즈오차) ~ 15%인경우
    train set error : 15% dev set error : 16% => 낮은 편향과 분산을 가진다
    ex)흐릿한 이미지

## 머신러닝 성능을 쳬계적으로 향상시키는 방법
1차 모델 훈련 -> 높은 편향을 가지는지 확인 : 훈련 set 데이터를 확인
       경우1. 높은 편향을 가지는 경우 : 더 많은 은닉층 사용 or 더 많은 은닉유닛 사용 or 더 오래 훈련 or 다른 최적화 알고리즘 사용 (or 다른 신경망 아키텍처 사용 - 작동할수도, 안할수도 있음) -> 모델 훈련부터 다시 시작(반복)
       경우2. 수용가능한 편향을 가지는 경우 -> 분산문제가 있는지 확인
              경우 2-1. 높은 분산을 가지는 경우 : 더 많은 데이터 이용 or 정규화 (or 다른 신경망 아키텍저 - 작동할수도, 안할수도 있음) -> 모델 훈련부터 다시 시작(반복)
              경우 2-2. 적당한 분산을 가지는 경우 -> 끝

## 편향-분산 트레이드 오프
과거(딥러닝 이전) : 시도할 수 있는 많은 툴이 없었다
                    편향이 오르고 분산이 내리거나
                    편향이 내리고 분산이 오르거나
현재(딥러닝 시대) : 많은 데이터를 가지고도 편향은 유의미한 변화가 없는데 분산이 내려가는 등 편향과 분산이 서로 영향을 주지 않고 작아지는 방법이 존재하다
                    -> 딥러닝이 유용하다 (가끔 어쩔 수 없이 편향이 오르고 분산이 내려가거나, 편향이 내리고 분산이 올라가는 경우가 존재)

## 정규화
높은 분산(데이터 과대적합)인 경우 분산을 낮춰주기 위한 방법
lambd : 정규화 매개변수 : 두매개변수의노름을 잘 설정해 과대적합을 막을 수 있는 최적의 값 탐색
        설정이 필요한 하이퍼 파라미터
        lambda는 파이썬 함수의 이름이기 때문에 이와 혼동하지 않기 위해 lambd로 표기
로지스틱회귀에서 L2정규화 : J(w, b) = 1/m * L(y_hat^(i), y^(i)) + lambd/2m * ||w||^2(sub)2 + (lambd/2m * b^2)-보통생략
w : x차원 벡터
b : 실수
||w||^2(sub)2 : 유클리드 노름의 제곱
w : 꽤 높은 차원의 매개변수 벡터, 분산이 높을수록 많은 매개변수를 포함
b : 하나의 실수 -> 넣어도 실질적인 변화가 없음 => 보통 생략한다

L1정규화 : lambd/2m * ||w||^2(sub)2  대신 lambd/m * S(i=1부터 n_x까지)|w| = lambd/m ||w||1 
w는 희소하다(w안에 0이 많다) -> 모델 압축에 도움이 되지만 실질적으로 큰 도움이 되지 않는다
=> L2에 비해 잘 사용하지 않는다

신경망에서 L2 정규화 : J(w^[1], b^[1],...,w^[L], b^[L]) = 1/m S(i=1부터 n까지)L(y_hat^(i), y^(i)) + 1/2m S(l=1부터 L까지)||w^[l]||^2(sub)F
||w^[l]||^2(sub)F : 프로베니우스 노름 : 행렬 원소 제곱의 합
w : (n^[l-1], n^[l]) : l-1과 l의 은닉유닛 개수
-> 경사하강법
dw역전파 : dJ/dw^[l] + lambd/m * w^[l]
w^[l] := w^[l] - adw^[l] 
       = w^[l]a[(역전파로 얻은 것) + lambd/m * w^[l]]
       = w^[l] - alambd/m w^[l] -a(역전파로 얻은 것)
       = w^[l](1-alambd/m)
       = w^[l] - alambd/m * w^[l]
       => w^[l]이 어떤 값이든 값이 약간 더 작아진다 => L2정규화 : 가중치 감쇠