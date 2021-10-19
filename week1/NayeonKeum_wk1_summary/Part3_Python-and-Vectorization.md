
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

