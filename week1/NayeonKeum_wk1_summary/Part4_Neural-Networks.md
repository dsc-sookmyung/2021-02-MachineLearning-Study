
# 4. NNs
***
#### NNs Representation
* 2 Layer NN(doesn't count the input layer)
<img src="https://user-images.githubusercontent.com/68985625/135476032-d2228175-c97d-44a6-accc-5c49fa33d511.png">


#### Computing NN Output
* Consists of
<img src="https://user-images.githubusercontent.com/68985625/135476376-68c8d724-3e87-4196-a579-d8c91d73545b.png">

<img src="https://user-images.githubusercontent.com/68985625/135476644-71b20bc2-e82d-4570-98ce-90da7acfcd6a.png">


#### Vectorizing Across Multiple Ex
<img src="https://user-images.githubusercontent.com/68985625/135477447-4563b15d-88aa-4b2c-bea1-629ffdc88016.png">
* matrix a (similar to z) : different hidden units * training examples
<img width="30%" src="https://user-images.githubusercontent.com/68985625/135477824-0d66ad42-7d7a-470d-b6ca-5dedd7a53cd4.png">


#### Explanation For Vectorized Implementation
* 

#### Activation Functions
* Use in hidden layers and output units
<img src="https://user-images.githubusercontent.com/68985625/135481282-6670b3d4-df6b-49ea-ae39-c39889eb2383.png">
<img src="https://user-images.githubusercontent.com/68985625/135481307-07a4cc54-c17e-4637-8ce0-ace5de7f406e.png">



* Sigmoid (0, 1)
<img src="https://user-images.githubusercontent.com/68985625/135479187-8aa83c05-8cc0-4cbc-bb0a-2ca605e541d6.png">

* Tanh (-1, 1)
<img src="https://user-images.githubusercontent.com/68985625/135479335-c8b7d7dc-4809-4b49-aa64-9fb5c2653c9c.png">
  - almost always better than Sigmoid with hidden units
    + result of the activation from the hidden layer are close to having a 0 mean
    + you might need to **center** your data using it
  - except for the output layer(for a binary classification)
    + if y is 0 or 1, y_hat can be more likely between 0 and 1(which is Sigmoid rather than Tanh)


* ReLU
<img src="https://user-images.githubusercontent.com/68985625/135480399-7f9a1daf-f2f1-43c9-92df-ee5adc43c6e0.png">
  - increasing defalut choice
  - advantage : the slop(기울기/도함수) is very diffrent from 0, learns much more faster
    + closer the slope to 0, slower the training
  - disadvantage : derivative(도함수) is 0 when z is negative
    + -> leaky ReLU

* leaky ReLU
<img src="https://user-images.githubusercontent.com/68985625/135481059-b751dac2-6643-42b8-9b94-33257b26601d.png">
  - why is it 0.01z?
    + feel free to try changing it, but no one trys it


#### Why Non-linear Activation Functions 
* By using a linear activation function, always turns out to be a linear function whether you stack(composite) it or not.
* Can't take any advantages stacking the layers

#### Derivatives Of Activation Functions 
<img src="https://user-images.githubusercontent.com/68985625/135482617-d5f3cb54-75a9-4ca7-8d26-3e39bbbc7b81.png">
<img src="https://user-images.githubusercontent.com/68985625/135482544-42e149bf-a3be-47b4-b636-dec7fa00a937.png">


#### Gradient Descent For Neural Networks
<img src="https://user-images.githubusercontent.com/68985625/135485041-d64de4bf-f1b6-4a61-ae83-346d93a30f13.png">
(prove yourself if possible)

#### Backpropagation Intuition
<img src="https://user-images.githubusercontent.com/68985625/135485212-de11bc20-bfaa-4ea2-b69e-f5d33898e00f.png">
(prove yourself if possible)

#### Random Initialization
* if NN is inititialized all in 0 and step to gradient descent, it won't work
  + both hidden units start computing the same function
  + influence same result to the next units
  + in this case, there's no point to have units because they compute the same thing
* instead, set W random
  - b doesn't have any symetry breaking probs beacuse of initialized w
```{.python}
w_i = np.random.randn((2, 2))*0.01
b_i = np.zero((2, 1))
```
  - why 0.01?
    + needs small value
    + if too large, values can be either very large or very small
    + makes gradient descent very slow -> training slow too
  - if w is too large
    -> more likely to start training with a large z
    -> causes tanh or sigmoid function tp be exaggerated
    -> thus slowing down learning
  - if you don't use tanh or sigmoid its less issue, but if you're doing binary classification or using sigmoid/tanh as the output unit, you won't want the initial parameters too large


***

