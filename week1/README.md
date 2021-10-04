## Deep Learning Basics
from [Lecture](https://www.edwith.org/ai215/joinLectures/86246?isDesc=false "Andrew Ng Deep Learning Basics")

# 1. Introduction to Deep Learning    
***
#### AI=Electricity       
* effecting almost every industry    

#### What is a Neural Network?   
* single neuron : x --(node-function)--> y       
* multi neurons : x1, x2, ... xn --(nodes-functions)--> y    

* Neural Network :    
  - every features are connected to every nodes(first layer)    
  - nodes(hidden units) and functions each(ex RELU)    
  - *given enough train sets, NNs are remarkably good at figuring out (activtation)functions* that accurately map from X to y    
  - powerfull in Supervised learning    
  
* Supervised Learning    
  - Xs, Ys and Applications    
  - examples    
    + Standard NN(nodes and funcs0    
    + Convolution NN(images etc)    
    + Recurrent NN(sequentail data like nlp etc)    
  - Data    
    + Structured Data : each features have very well-defined meaning    
    + Unstructured Data(Audios, Images, Text etc) : features are pixels, individual words    
      + compared to Structured data, humans have evolved to be good at interpreting *unstructured data*    
      + thanks to NN, computers are now much better than few years ago    
    - Still, economic benefits come from Structured data actually    

* Drivers of DL    
  1. Increased Scale    
    - Scales of Performance for the Amount of data    
      1. Traditional learning alog(SVM, LR etc) : limit to the increase(turns to horizontal increments)    
        + don't know what to do with huge amounts of data from nowadays    
      2. NN : as more and more data is put, performance keep gets better and better    
      -> In order to increase performance, NN *needs a lot of LABELED data* and *a big size(hidden units+parameters+connections)*    
      + If small training sets, there isn't and order between 1 and 2    

  2. Increased computation    
  
  3. Improvement of Algorithms    
    - Gradient Descent got faster changing Sigmoid -> RELU    
      + Sigmoid : slow training regions where gradients are almost zero    
      + RELU(Rectified Linear Unit) : gradients equal to 1 in all part of the input, so it doesn't gradually head to zero(rather momentarily)    


  - with 2 and 3, training time reduced from 1 months to 10 mins, days(not always) so it became much more likely to discover NN that works well for applications    
  
***


# 2. Basics of Neural Network Programming    
***
#### Binary Classification      
* Image
  - pixel(c*r)* 3(rgb)
  - 0 or 1
  - Sigmoid( -infinite approximit 0, infinite to 1)
    + y(z)=1/(1+e^(-z))

#### Logistic Regression Cost Function
* Loss (error) function : measures how good our y_hat is compared to y(true label)(single training example)
  - doesn't work well in GD
  * L(y^, y)=-(ylogy^+(1-y)log(1-y^))
  * if y=1 : L(y^, y)=-logy^ (y^ large)
  * if y=0 : L(y^, y)=-log(1-y^) (y^ small)
* Cost function : measures how good our w, b are doing in our entire training example
  * J(w, b)=1/m(sigma L(y^, y))=-1/m(sigma ylogy^+(1-y)log(1-y^))

#### Gradient Descent
* train parameters w and b
* finding w and b which makes J(w, b) as small as possible(b is dimension)
* w := w - a * dJ(w, b)/dw
* b := b - a * dJ(w, b)/db

#### Computation Graph
<img src="https://user-images.githubusercontent.com/68985625/134808546-0085f341-dc06-4118-8b12-fc4dd5bd2054.png">


***


# 3. 
***
#### What is vectorization
* z=w^(t)*x + b
  - ```{.python}
    z = np.dot(w, x)+b```
  - CPU vs GPU?
    + SIMD(Single Instruction Multiple Data)
    + if using built-in function, it enables to take much better advantage due to parellelism
    + GPU is faster than CPU
  - **avoiding for-loops is very important!!**

#### NN programming guideline
* Whenever possible, avoid explicit for-loops
  - in LR derivatives, can use vectorization in computing dws

#### Vectorizing LR
* ```{.python}
  z = np.dot(w.T, x)+b```

#### LR Gradient Descent
* if use for-loop

<img src="https://user-images.githubusercontent.com/68985625/135099140-5728e46e-ad46-4cf7-856d-81665cd01924.png">

* if use vector
<img src="https://user-images.githubusercontent.com/68985625/135099213-22a7a8dc-e8d4-4259-b194-8f96700f81b7.png">


#### Broadcasting in python
* python code(ref to Part3_Python-and-Vectorization.ipynb)
```{.python}
A=np.array([[56.0, 0.0, 4.4, 68.0],
            [1.2, 104.0, 52.0, 8.0],
            [1.8, 135.9, 99.0, 0.9]])

print(A)
```
[[ 56.    0.    4.4  68. ]
 [  1.2 104.   52.    8. ]
 [  1.8 135.9  99.    0.9]]

```{.python}
cal=A.sum(axis=0)
print(cal)
```
[ 59.  239.9 155.4  76.9]

```{.python}
percentage=100*A/cal#.reshape(1, 4)
print(percentage)
```
[[94.91525424  0.          2.83140283 88.42652796]
 [ 2.03389831 43.35139642 33.46203346 10.40312094]
 [ 3.05084746 56.64860358 63.70656371  1.17035111]]


* Broadcasting example
  - [ [1], [2], [3], [4]] + 100 = [[101], [102], [103], [104]]
  - [[1, 2, 3], [4, 5, 6]] + [100, 200, 300] = [[101, 202, 303], [104, 205, 306]]
  - [[1, 2, 3], [4, 5, 6]]+[[100], [200]] = [[101, 102, 103], [204, 205, 206]]


#### Python/Numpy Vectors
* don't use
  - ```{.python}
    a=np.random.randn(5)
    ```
  - rank 1 array
* instead use
  - ```{.python}
    a=np.random.randn(5,1)
    ```
  - a.shape=(5,1)


  - ```{.python}
    a=np.random.randn(1,5)
    ```
  - a.shape=(1,5)

* assertion statement
  - ```{.python}
    assert(a.shape==(5,1))
    a=a.reshape((5,1))
    ```

***

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
  - advantage : the slop(±â¿ï±â/µµÇÔ¼ö) is very diffrent from 0, learns much more faster
    + closer the slope to 0, slower the training
  - disadvantage : derivative(µµÇÔ¼ö) is 0 when z is negative
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

# 5. Deep NN
***

#### Deep L-Layer Neural Network
* 
<img src="https://user-images.githubusercontent.com/68985625/135755394-156a32f4-4b4e-4bbf-9744-58782965ba25.png">

L=layers
n^[l]=units in layer l
a^[l]=activations in layer l
a^[0]=# of input units
a^[L]=# of output units

#### Forward and Backward Propagation
* building f/b propagation
<img src="https://user-images.githubusercontent.com/68985625/135755904-f051dcfb-b992-4b7f-ad30-dfa0de13fea4.png">


#### Forward Propagation in a Deep Network
* proof
<img src="https://user-images.githubusercontent.com/68985625/135755988-706d91f9-a91c-413a-afc5-3279bf57bcd2.png">


#### Getting Matrix Dimensions Right
* 
<img src="https://user-images.githubusercontent.com/68985625/135756994-9db8e2ec-b370-4625-ace4-97dc818655db.png">
<img src="https://user-images.githubusercontent.com/68985625/135757041-f13ddbd9-267b-44c4-898c-93b5e861940f.png">


#### Why Deep Representations? 
* Big? (Deep or Many hidden layers)
* Intuition about deep representation : Drill down detection
  - Simple(small) -> Complex(large) : Convolution NN
* Circuit theory and deep learning
  - Informally, There are functions you can compute with a "small" L-layer deep neural network that shallower networks require exponentially more hidden units to compute
  - With Multiple Hidden Layers : network depth will be O(log n) : don't need lot of gates
  - If Not, : in order to compute XOR functions, hidden layers should be **exponentially** large (in a number of bits)


#### Building Blocks of a Deep Neural Network
<img src="https://user-images.githubusercontent.com/68985625/135757676-406e15c5-ac22-4c6c-afee-276c62138cbd.png">
<img src="https://user-images.githubusercontent.com/68985625/135757709-bf952c91-7e73-466d-b0b7-53d7a1eb7e5e.png">


#### Parameters vs Hyperparameters
* Parameters : W^[1], b^[1], W^[2], ...

* Hyperparameters : Parameters that control the ultimate(above) parameters
  - Determine the final values W and b
  - learning rate alpha
  - iterations
  - hidden layers L
  - hidden unit n^[1], n^[2], ...
  - choice of activation function

* Applying Deep Learning..
  - Serendipity(empirical, heuristic)
  - impliment, try it, change it ...
    + See the cost function change by replacing Hyperparameters
  - **just try out a range of values**(will learn systematic version soon)
  - even if you tuned the best values, it would change 1 year after(due to the change of computer infrastructure like CPUs, type of GPUs)


#### What does it have to do with the brain?
* easy to understand, say it publicly, easy for the media to report it
* Has similar logic :input-output mapping
* especially CV has been inspired from the human brain



***






