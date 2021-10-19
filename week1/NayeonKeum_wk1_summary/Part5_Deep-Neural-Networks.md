
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

