# 1. Setting up your ML application
***

#### Train/Dev/Test Sets
* Making choices in how you develop your test sets can make a huge difference in helping you quickly find a good performance in NN
* Lot of decisions
    - #layers
    - #hidden units
    - learning rates
    - activation functions
    - etc
* Idea -> Code -> Experiment -> Idea -> ...
* Intuitions do not transfer to other application areas -> best choices depend on data size, # of input features, gpus and cpus
* Impossible to correctly guess the right parameters
* **How efficiently you can go around the cycle**
* Keep training **training set** -> use **development set** to see which model performs best -> train -> dev -> ... -> evaluate your final model by **test set**  in order to get a unbiased estimate/how well yout algo is doing
* conventionaly 60/20/20 but ratio can change due to the size of data
* Mismatched train/test distribution
    - Training set: Cat pictures from sebpages
    - Dev/test sets: Cat pictures from users using you app
    - to distributions of data can be different
    - -> Make sure dec and test com from the *same distribution*
* Not having a test set might be okay. (Only dev set)



#### Bias/Variance 
* Bias/Variance Tradeoff
<img src="https://user-images.githubusercontent.com/68985625/136127887-6c528cd8-0080-4998-b0c1-2230ff4dbccf.png">

* Optimal(Begies) error is the Benchmark of train/dev/test errors
* example1
    - (asume) human error / Optimal(Begies) error : 0%
    - Train set error : 1%
    - Dev set error : 11%
    - -> high variance(overfitting / poor cross validation)
* example2
    - (asume) human error / Optimal(Begies) error : 0%
    - Train set error : 15%
    - Dev set error : 16%
    - -> high bias(underfitting / doesn't even fit the training set)
* example3
    - (asume) human error / Optimal(Begies) error : 0%
    - Train set error : 15%
    - Dev set error : 30%
    - -> high variance / high bias
* example4
    - (asume) human error / Optimal(Begies) error : 0%
    - Train set error : 0.5%
    - Dev set error : 1%
    - -> low variance / low bias


#### Basic Recipe for Machine Learning 
* (after training) 1st question. High bias?
    - training data performance
    - (yes) Bigger network / Train longer / (NN architecture search - more appropriate) -> until getting rid of the bias
    - (no) good.
* 2nd. High variance?
    - dev set performance
    - (yes) More data / Regularization / (NN architecture search - more appropriate)
    - (no) good.
* Bias variance tradeoff
    - modern bigdata era, getting a bigger network/getting more data reduces bias/variance that don't hurt the other thing
    - -> Regualization can help

***

