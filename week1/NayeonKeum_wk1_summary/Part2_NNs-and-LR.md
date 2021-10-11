
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

