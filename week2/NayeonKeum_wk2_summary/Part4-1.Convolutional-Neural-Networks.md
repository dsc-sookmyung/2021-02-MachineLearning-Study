# 7. Convolutional Neural Networks
***

#### Computer Vision
* 1. Brand new applications
* 2. Computer Vision Research community itself is creative and inventive influencing other areas

* Computer vision problems
    - Image classification
    - Object detection
    - Nerual Style Transfer(Repainting with a different style; new artwork)

* Challenges
    1. images are too large
        - every pixel of an image is an input feature, and there's three rgb channel so its very large



#### Edge Detection Examples
* detect vertical and horizontal edges
* How to detect it?
    - <img src="https://user-images.githubusercontent.com/68985625/136507188-eb7ba3e1-c914-4ba8-af2a-e3737e650aad.png">
    - python: conv-forward
    - tensorlow: tf.nn.conv2d
    - keras: Conv2D
    - <img src="https://user-images.githubusercontent.com/68985625/136507334-148f02f7-8801-442f-a17e-a05c147a3178.png">
    - filter can specify vertical edges


#### More Edge Detection
* Vertical and Horizontal Edge Detection
<img src="https://user-images.githubusercontent.com/68985625/136507747-09978910-5648-462e-b0d6-af43cd73cc12.png">
* Many kinds of filters
<img src="https://user-images.githubusercontent.com/68985625/136507443-7c9b1508-525b-422a-86c2-0bbe85094c5c.png">
    - can detect whatever orientation it chooses


#### Padding
* Why padding?
    1. image shrinks while detecting edges or other features
        - end up with a very small image
    2. pixels on the corners are not used that much
    -> before convolutional operation, pad image an additional border

* Valid and Same convolution
    - result size: (n + 2p - f + 1) x (n + 2p - f + 1)
    - n: size of image
    - p: size of padding
    - f: size of filter
    - p=(f-1)/2 (f is almost always odd)


#### Strided Convolutions
* striding is jumping postition while filtering
* result size: ((n + 2p - f)/s + 1) x ((n + 2p - f)/s + 1)
<img src="https://user-images.githubusercontent.com/68985625/136508913-7904bc82-e1b9-4e22-8ec1-d2ffea7ea969.png">
    - work throw n, p, f, s

#### Convolutions Over Volumes
* result in multi-dimension
<img src="https://user-images.githubusercontent.com/68985625/136509142-fc546c2e-c320-43db-b1a5-67fe83e3028d.png">
<img src="https://user-images.githubusercontent.com/68985625/136509251-3a679f51-9c73-4860-b3ea-d304dcf4f1dc.png">


#### One Layer of a Convolutional Net
<img src="https://user-images.githubusercontent.com/68985625/136509500-624fdf20-2bbe-4c27-b15b-21193a3557ea.png" width="30%">
<img src="https://user-images.githubusercontent.com/68985625/136509615-bd704cad-37a1-45fd-8689-033123d14e9b.png">

#### Simple Convolutional Network Example
* example
<img src="https://user-images.githubusercontent.com/68985625/136509687-26f71030-d4d1-4c48-8132-17a011c49151.png">

* Types of layer
    1. Convolutional layer(CONV)
    2. Pooling (POOL)
    3. Fully connected (FC)

#### Pooling Layers
<img src="https://user-images.githubusercontent.com/68985625/136510019-411bd805-dea9-44f0-a702-24a13ec8542c.png">

* hyperparameters
    - f: filter size
    - s: stride

* Max pooling
    - need to set up hyperparameters, but they don't learn(no influence in Gradient Descent) 
    - usually **do not use padding**
* Average pooling
    - don't take the max value, instead the average one


#### CNN Example
* example
<img src="https://user-images.githubusercontent.com/68985625/136510461-37222c51-c166-4900-a5fd-1c217ece7ae1.png">
    - two types of conventions in the field of convolutional neural networks: one sees a convolutional layer and a pooling layer as one layer, and the other considers the convolutional layer and the pooling layer as each layer. 
    - we'll use the former method
    - Since there are no variables to learn in the pooling layer, the convolutional and pooling layers are considered one.
<img src="https://user-images.githubusercontent.com/68985625/136510648-cb85f112-0aed-4dcd-b1a2-a172d693a764.png">

* POINT
    1. Max pooling layers don't have any params
    2. conv layers have few params(tend to be in the fully-connected alyer)
    3. size reduces gettind deeper


#### Why Convolutions
* Parameter sharing 
    - A feature detector (such as a vertical edge detector) that's useful in one part of the image is probably useful in another part of the image 
    -> reduces the number of parameters
* Sparsity of conntections
    - In each layer, each output value depends only on a small numver of inputs
    - can prevent overfitting
* Easy to capture movement invariance  
    - even if the image is slightly deformed, it can be captured.


***
