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



