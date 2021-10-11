# 2. Regularizing your neural network
***

#### Regularization
* L2 Regularization : Regularize W in LR
    - W has a lot of parameters(b is just a number)
    <img src="https://user-images.githubusercontent.com/68985625/136130194-072f45e0-eb25-4013-af9b-f6d43471e942.png">
* L1 Regularization : Regularize W in LR
    - W has a lot of parameters(b is just a number)
* NN
    - <img src="https://user-images.githubusercontent.com/68985625/136130660-09dba835-7ef2-4391-ae50-087ff33adc8a.png">

#### Why Regularization Reduces Overfitting 
* Use every unit but each effection reduces(end up with a simpler network) -> can reduce overfitting
* variance reduction
<img src="https://user-images.githubusercontent.com/68985625/136131048-0a2816a5-13e4-4756-80a7-daf2db618fa3.png">

#### Dropout Regularization
* eliminating nodes randomly
* training smaller networks in each layers
* Inverted dropout
    - layer = 3
    - randomly zero-out units
* Making predictions at test time
    - no drop out
    - even if you don't dropout at test time the expected values of activation function doesn't change

#### Understanding Dropout 
* Why does drop-out work?
    - Intuition: *Can't rely* on any one fearue, so have to spread out weights
    - -> shirnks weights(similar to L2 Regularization)
* keep-prob : kepp going probability
    - not using drop-outs
    - number close to 1.0 is common
* CV
    - input pixels(size big) -> almost never have enough data
    - drop-out is frequently used
* Conclude, dropout is a kind of Regularization technic and prevents overfitting. So, unless my algo is overfitting, I wouldn't bother using dropout

#### Other Regularization Methods 
* Data augmentation
    - For images, we can solve overfitting by using more training data.
    - We usually create new training data by mirroring, magnifying, distorting, or rotating the image.
    - These additional fake images don't add more information than getting a completely new, independent sample, but they do have the advantage that they can be done at no computational cost.

* Early stopping
    - The error in the training set will be plotted as a monotonic descent function.
    - Early shutdown also draws errors in the development set.
    - If the error of the development set starts to increase rather than decrease at some point, it is the time of over-optimization.
    - -> premature termination is when the neural network stops training near the error trough of the development set, the point at which it works best.
    - Disadvantages: During training, there is a task of optimizing the cost function(training objective) and a task of making it not overfitting. The two tasks are separate things and must be approached in two different ways. But early termination mixes the two. 
    -> it may not be possible to find optimal conditions.


***