## GDSC
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
