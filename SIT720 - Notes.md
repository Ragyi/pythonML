
# SIT 720 - Python Intro


```python
dic = {}
dic['one'] = 'This is one'
dic[2] = 'This is two'
print(dic['one'])
print(dic)
dic['one'] = 'One has changed'
print(dic)
```

    This is one
    {'one': 'This is one', 2: 'This is two'}
    {'one': 'One has changed', 2: 'This is two'}



```python
l = [1, 2, 3, 4, 5, 6, 7, 8]
s = "This is a string."

print(l[1:5])
print(s[0:6])
```

    [2, 3, 4, 5]
    This i


#### Branching and Decisioning


```python
trip1 = 15

if trip1 <= 25:
    print('25 and Under')
else:
    print("25 and Over")
```

    25 and Under



```python
stat1 = True
stat2 = False

if stat1:
    print("1st stat is true")
elif stat2:
    print("2nd stat true")
else:
    print("Both are false")
```

    1st stat is true


#### Iterations (Loops)

##### For Loops


```python
exampleList = [1, 2, 3, 4, 5]

for i in exampleList:
    print(i)
```

    1
    2
    3
    4
    5



```python
#String with dynamic object
x=list(range(2,6))

print("Initial List: {}".format(x))

for idx, i in enumerate(x):
    x[idx]= i**2
    
print("The new list:{}".format(x))

#During each step of the for loop, enumerate(x)iterates through the list 
#and store the index in [idx] and value in [i].
```

    Initial List: [2, 3, 4, 5]
    The new list:[4, 9, 16, 25]



```python
newList = [x**2 for x in range (2, 6)]
print(newList)
```

    [4, 9, 16, 25]


##### While Loops


```python
i = 0
while i < 5:
      print (i, end=" ")   # prints each iteration on the same line
      i += 1            # adds 1 to the variable i
print()                 # prints a blank line
print("done")       # Note that this is printed outside the loop
```

    0 1 2 3 4 
    done



```python
y = range(1, 51)

for i in y: 
    if i%3==0 and i%2!=0:
        print(i)
```

    3
    9
    15
    21
    27
    33
    39
    45


##### Functions


```python
def func1(s):
    "Some stuff"
    
    print("Number of characters in the string: ", len(s))
    return 2*len(s)
```


```python
func1("test function")
```

    Number of characters in the string:  13





    26



###### Returning multiple values from a function


```python
def powers(x):
    xs = x**2
    xc = x**3
    xf = x**4
    return xs, xc, xf
```


```python
powers(5)
```




    (25, 125, 625)




```python
y1, y2, y3 = powers(5)
print(y2)
```

    125


##### Anonymous Functions

Anonymous functions are defined by the keyword  lambda  in Python. Functions  f  and  g  in the cell below basically do the same thing. But  g  is an anonymous function.


```python
# define a function
def f(x):                 
    return x**2.          # x to the power of 2 - the function give us x squared

# use an anonymous function instead
g = lambda x: x**2.  # x to the power of 2 - in other words x squared

print(f(8))   # call the f function and ask for the square of 8
print(g(8))  # call the g anonymous function and ask for the square of 8
```

In the cell below, we used anonymous function n_increment(). We create new functions by passing  n  to n_incremenet(). For example  f5  and  f9  are functions that add 5 and 9 to their inputs respectively.


```python
def n_increment(n):
    return lambda x: x+n
add5 = n_increment(5)
print(add5(2))
```

    7


### Distances

![](cosine.png)

#### Manhattan Similarity


```python
from math import*
 
def square_rooted(x):
 
    return round(sqrt(sum([a*a for a in x])),3)
 
def cosine_similarity(x,y):
 
    numerator = sum(a*b for a,b in zip(x,y))
    denominator = square_rooted(x)*square_rooted(y)
    return round(numerator/float(denominator),3)
 
print (cosine_similarity([1,1,1], [3,3,3]))
```

    1.0


#### Jaccard Similarity


```python
from math import*
 
def jaccard_similarity(x,y):
 
    intersection_cardinality = len(set.intersection(*[set(x), set(y)]))
    union_cardinality = len(set.union(*[set(x), set(y)]))
    return intersection_cardinality/float(union_cardinality)
 
print (jaccard_similarity(["banana","orange","grapes"],["apple", "banana", "grapes", "pear"]))
```

    0.4


#### Eucldean Distance


```python
from math import *
```


```python
v1 = [1,1,1]
v2 = [3,3,3]
v3 = [3,0,2,6]
```


```python
def eclu_dist(x,y):
    return sqrt(sum(pow(a-b, 2) for a,b in zip(x,y)))
```


```python
print(eclu_dist(v1,v2))
```

    3.4641016151377544


#### Manhatan Distance


```python
def man_dist(x,y):
    return sum(abs(a-b) for a,b in zip(x,y))
```


```python
print(man_dist(v1,v2))
print(man_dist(v1,v3))
```

    7
    10



```python
c1 = [1,2]
c2 = [4,3]
c3 = [4,1]
```


```python
pt = [2,2]
```


```python
print(man_dist(pt, c1))
print(man_dist(pt, c2))
print(man_dist(pt, c3))
```

    1
    3
    3


#### Intro to Numpy Module


```python
import numpy as np
```


```python
a = np.random.randn(5,1)
print(a.shape)
print(a)
```

    (5, 1)
    [[ 0.02065934]
     [-0.63937074]
     [-0.20289277]
     [-0.26456445]
     [ 0.70719702]]



```python
a_trans = a.T
print(a_trans.shape)
print(a_trans)
```

    (1, 5)
    [[ 0.02065934 -0.63937074 -0.20289277 -0.26456445  0.70719702]]



```python
a_dot = np.dot(a, a_trans)
print(a_dot.shape)
print(a_dot)
```

    (5, 5)
    [[ 4.26808391e-04 -1.32089784e-02 -4.19163093e-03 -5.46572735e-03
       1.46102247e-02]
     [-1.32089784e-02  4.08794938e-01  1.29723697e-01  1.69154768e-01
      -4.52161079e-01]
     [-4.19163093e-03  1.29723697e-01  4.11654743e-02  5.36782133e-02
      -1.43485159e-01]
     [-5.46572735e-03  1.69154768e-01  5.36782133e-02  6.99943491e-02
      -1.87099191e-01]
     [ 1.46102247e-02 -4.52161079e-01 -1.43485159e-01 -1.87099191e-01
       5.00127623e-01]]


#### Installing Packages in Python 


```python
##Using Pip Install
import sys
!{sys.executable} -m pip install tensorflow
```

    Requirement already satisfied: tensorflow in ./anaconda2/envs/ML/lib/python3.7/site-packages (1.14.0)
    Requirement already satisfied: astor>=0.6.0 in ./anaconda2/envs/ML/lib/python3.7/site-packages (from tensorflow) (0.8.0)
    Requirement already satisfied: keras-preprocessing>=1.0.5 in ./anaconda2/envs/ML/lib/python3.7/site-packages (from tensorflow) (1.1.0)
    Requirement already satisfied: termcolor>=1.1.0 in ./anaconda2/envs/ML/lib/python3.7/site-packages (from tensorflow) (1.1.0)
    Requirement already satisfied: keras-applications>=1.0.6 in ./anaconda2/envs/ML/lib/python3.7/site-packages (from tensorflow) (1.0.8)
    Requirement already satisfied: tensorboard<1.15.0,>=1.14.0 in ./anaconda2/envs/ML/lib/python3.7/site-packages (from tensorflow) (1.14.0)
    Requirement already satisfied: tensorflow-estimator<1.15.0rc0,>=1.14.0rc0 in ./anaconda2/envs/ML/lib/python3.7/site-packages (from tensorflow) (1.14.0)
    Requirement already satisfied: absl-py>=0.7.0 in ./anaconda2/envs/ML/lib/python3.7/site-packages (from tensorflow) (0.7.1)
    Requirement already satisfied: google-pasta>=0.1.6 in ./anaconda2/envs/ML/lib/python3.7/site-packages (from tensorflow) (0.1.7)
    Requirement already satisfied: six>=1.10.0 in ./anaconda2/envs/ML/lib/python3.7/site-packages (from tensorflow) (1.12.0)
    Requirement already satisfied: wheel>=0.26 in ./anaconda2/envs/ML/lib/python3.7/site-packages (from tensorflow) (0.33.4)
    Requirement already satisfied: protobuf>=3.6.1 in ./anaconda2/envs/ML/lib/python3.7/site-packages (from tensorflow) (3.8.0)
    Requirement already satisfied: numpy<2.0,>=1.14.5 in ./anaconda2/envs/ML/lib/python3.7/site-packages (from tensorflow) (1.16.4)
    Requirement already satisfied: wrapt>=1.11.1 in ./anaconda2/envs/ML/lib/python3.7/site-packages (from tensorflow) (1.11.2)
    Requirement already satisfied: grpcio>=1.8.6 in ./anaconda2/envs/ML/lib/python3.7/site-packages (from tensorflow) (1.22.0)
    Requirement already satisfied: gast>=0.2.0 in ./anaconda2/envs/ML/lib/python3.7/site-packages (from tensorflow) (0.2.2)
    Requirement already satisfied: h5py in ./anaconda2/envs/ML/lib/python3.7/site-packages (from keras-applications>=1.0.6->tensorflow) (2.9.0)
    Requirement already satisfied: setuptools>=41.0.0 in ./anaconda2/envs/ML/lib/python3.7/site-packages (from tensorboard<1.15.0,>=1.14.0->tensorflow) (41.0.1)
    Requirement already satisfied: werkzeug>=0.11.15 in ./anaconda2/envs/ML/lib/python3.7/site-packages (from tensorboard<1.15.0,>=1.14.0->tensorflow) (0.15.4)
    Requirement already satisfied: markdown>=2.6.8 in ./anaconda2/envs/ML/lib/python3.7/site-packages (from tensorboard<1.15.0,>=1.14.0->tensorflow) (3.1.1)


#### Using MatPlot in Pythong


```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
```

Create to arrays X and Y


```python
x = np.array([1,2,3,4])
y = np.array([10,12,33,56])

xx = np.array([1.5,2.5,3.5])
yy = np.array([10.5, 15.5, 20])
```

plot x and y


```python
plt.plot(x,y, '*r') # The plt.plot function takes the argument for plot type *r = red stars
plt.show() # By having a plt.show() for each plot this will create 2 seperate plots
plt.plot(xx,yy, '.b') # This will overlay the current plot with blue dots (.b)
plt.show() # This line shows the plot
```


![png](output_55_0.png)



![png](output_55_1.png)


#### Extending Numpy


```python
A = np.array([(1,2), (3,4)])
print(A)
```

    [[1 2]
     [3 4]]


An all zero matrix


```python
B = np.zeros([3,3])
print(B)
```

    [[0. 0. 0.]
     [0. 0. 0.]
     [0. 0. 0.]]


All 1 matrix


```python
C = np.ones([3,3])
print(C)
```

    [[1. 1. 1.]
     [1. 1. 1.]
     [1. 1. 1.]]


Identity Matrix


```python
D = np.identity(3)
print(D)
```

    [[1. 0. 0.]
     [0. 1. 0.]
     [0. 0. 1.]]


Matrix of random numbers


```python
E = np.random.randn(4,3)
print(E)
```

    [[-1.99030373 -1.13753217  0.57406588]
     [-0.53403094  0.68025265  0.55211424]
     [ 0.12718483  0.62083608 -1.46039388]
     [-1.08696216 -0.43184155  0.24903507]]


##### Adding or subtracting a scalar value to a matrix 


```python
print(A)
print()
print("After addition of a scalar: ")
print(A+3)
```

    [[1 2]
     [3 4]]
    
    After addition of a scalar: 
    [[4 5]
     [6 7]]


##### Adding or subtracting matrices


```python
aa = np.identity(2)
bb = np.random.randn(2,2)
print("Matrix AA")
print(aa)
print("Matrix BB")
print(bb)
```

    Matrix AA
    [[1. 0.]
     [0. 1.]]
    Matrix BB
    [[-0.90370294 -0.3111972 ]
     [-0.35184552  0.19191158]]


Lets add aa and bb together


```python
result = aa + bb
print(result)
```

    [[ 0.09629706 -0.3111972 ]
     [-0.35184552  1.19191158]]


##### Multiplying matrices


```python
ac = np.random.randn(3,3)
ca = np.random.randn(3,2)
```


```python
print(np.shape(ac))
print(np.shape(ca))
```

    (3, 3)
    (3, 2)



```python
print(ac.dot(ca))
print("+++++++++++++++++++++++++++++++++++++++")
print(np.dot(ac,ca))
```

    [[-3.32449661  3.56454976]
     [ 1.48532252 -2.68407261]
     [-2.17719238  1.78806864]]
    +++++++++++++++++++++++++++++++++++++++
    [[-3.32449661  3.56454976]
     [ 1.48532252 -2.68407261]
     [-2.17719238  1.78806864]]


The otherway around does not work as the coulmns of the first is not equal to the rows of the second


```python
print(np.dot(ca,ac))
```


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    <ipython-input-161-c54724c0e8ab> in <module>
    ----> 1 print(np.dot(ca,ac))
    

    ValueError: shapes (3,2) and (3,3) not aligned: 2 (dim 1) != 3 (dim 0)


![](inversematrix.png)


```python
cc = np.random.randn(3,3)
cc_inverse = np.linalg.inv(cc)
print("This is the original matrix:")
print(cc)
print("This is it's inverse:")
print(cc_inverse)
```

    This is the original matrix:
    [[-0.14424467 -0.82311042  0.17631817]
     [ 0.10232438  0.24796583 -0.33836574]
     [ 0.72756318 -0.40811641 -0.39661549]]
    This is it's inverse:
    [[-1.44027345 -2.42695693  1.43023331]
     [-1.25240731 -0.43294114 -0.18741003]
     [-1.35335603 -4.00658613  0.29517306]]


Now let's check if the condition holds up:


```python
print(np.dot(cc, cc_inverse)) #should produce an identity matrix
print("Which is also identical to: ")
print(np.dot(cc_inverse, cc))
```

    [[ 1.00000000e+00 -1.43583071e-16 -5.15807765e-17]
     [ 2.81786443e-17  1.00000000e+00  1.30809612e-17]
     [-1.51406610e-16  1.51720120e-16  1.00000000e+00]]
    Which is also identical to: 
    [[ 1.00000000e+00 -1.07924509e-17 -1.82598655e-16]
     [ 3.29573910e-17  1.00000000e+00 -2.04147114e-17]
     [ 1.87792607e-17  1.54536858e-17  1.00000000e+00]]


#### Transposing a Matrix


```python
AA = np.arange(6).reshape(3,2)
BB = np.arange(8).reshape(2,4)
print(AA)
print("===========")
print(BB)
```

    [[0 1]
     [2 3]
     [4 5]]
    ===========
    [[0 1 2 3]
     [4 5 6 7]]


Transpose of A


```python
print(AA.T)
```

    [[0 2 4]
     [1 3 5]]


A note: Let matrix  A  be of dimension  n×m  and let  B  be of dimension  m×p. Then  (AB)′=B′A′


```python
print(np.dot(AA,BB).T)
```

    [[ 4 12 20]
     [ 5 17 29]
     [ 6 22 38]
     [ 7 27 47]]



```python
print(np.dot(BB.T, AA.T))
```

    [[ 4 12 20]
     [ 5 17 29]
     [ 6 22 38]
     [ 7 27 47]]



```python
print(A)
print("This is the first column of the matrix A: ")
print(A[:,0])
```

    [[1 2]
     [3 4]]
    This is the first column of the matrix A: 
    [1 3]



```python
print(A[-1,1])
```

    4


Using logical checks to extract values from matrices:


```python
#give the element in the last column that is greater than 3
print(A[:,1]>3)
```

    [False  True]


Create a  12×2  matrix and print it out:


```python
A = np.arange(24).reshape(12,2)
print(A)
```

    [[ 0  1]
     [ 2  3]
     [ 4  5]
     [ 6  7]
     [ 8  9]
     [10 11]
     [12 13]
     [14 15]
     [16 17]
     [18 19]
     [20 21]
     [22 23]]



```python
for i in A:
    print(i)
```

    [0 1]
    [2 3]
    [4 5]
    [6 7]
    [8 9]
    [10 11]
    [12 13]
    [14 15]
    [16 17]
    [18 19]
    [20 21]
    [22 23]



```python
for j in A.T:
    print(j)
```

    [ 0  2  4  6  8 10 12 14 16 18 20 22]
    [ 1  3  5  7  9 11 13 15 17 19 21 23]


#### Find the minimum of a function


```python
import numpy as np
from scipy.optimize import fmin
import math
```


```python
## Define the function

def f(x):
    val = math.pow(x,2) +1
    return val

funMin = fmin(f,np.random.randn(1,1))
print(funMin)
```

    Optimization terminated successfully.
             Current function value: 1.000000
             Iterations: 18
             Function evaluations: 36
    [1.33226763e-15]


#### Scikit-Learn Package for ML


```python
import sys
!{sys.executable} -m pip install -U scikit-learn
```

    Requirement already up-to-date: scikit-learn in ./anaconda2/envs/ML/lib/python3.7/site-packages (0.21.2)
    Requirement already satisfied, skipping upgrade: joblib>=0.11 in ./anaconda2/envs/ML/lib/python3.7/site-packages (from scikit-learn) (0.13.2)
    Requirement already satisfied, skipping upgrade: numpy>=1.11.0 in ./anaconda2/envs/ML/lib/python3.7/site-packages (from scikit-learn) (1.16.4)
    Requirement already satisfied, skipping upgrade: scipy>=0.17.0 in ./anaconda2/envs/ML/lib/python3.7/site-packages (from scikit-learn) (1.3.0)



```python
from sklearn import datasets
```


```python
digits = datasets.load_digits()
print(digits)
```

    {'data': array([[ 0.,  0.,  5., ...,  0.,  0.,  0.],
           [ 0.,  0.,  0., ..., 10.,  0.,  0.],
           [ 0.,  0.,  0., ..., 16.,  9.,  0.],
           ...,
           [ 0.,  0.,  1., ...,  6.,  0.,  0.],
           [ 0.,  0.,  2., ..., 12.,  0.,  0.],
           [ 0.,  0., 10., ..., 12.,  1.,  0.]]), 'target': array([0, 1, 2, ..., 8, 9, 8]), 'target_names': array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), 'images': array([[[ 0.,  0.,  5., ...,  1.,  0.,  0.],
            [ 0.,  0., 13., ..., 15.,  5.,  0.],
            [ 0.,  3., 15., ..., 11.,  8.,  0.],
            ...,
            [ 0.,  4., 11., ..., 12.,  7.,  0.],
            [ 0.,  2., 14., ..., 12.,  0.,  0.],
            [ 0.,  0.,  6., ...,  0.,  0.,  0.]],
    
           [[ 0.,  0.,  0., ...,  5.,  0.,  0.],
            [ 0.,  0.,  0., ...,  9.,  0.,  0.],
            [ 0.,  0.,  3., ...,  6.,  0.,  0.],
            ...,
            [ 0.,  0.,  1., ...,  6.,  0.,  0.],
            [ 0.,  0.,  1., ...,  6.,  0.,  0.],
            [ 0.,  0.,  0., ..., 10.,  0.,  0.]],
    
           [[ 0.,  0.,  0., ..., 12.,  0.,  0.],
            [ 0.,  0.,  3., ..., 14.,  0.,  0.],
            [ 0.,  0.,  8., ..., 16.,  0.,  0.],
            ...,
            [ 0.,  9., 16., ...,  0.,  0.,  0.],
            [ 0.,  3., 13., ..., 11.,  5.,  0.],
            [ 0.,  0.,  0., ..., 16.,  9.,  0.]],
    
           ...,
    
           [[ 0.,  0.,  1., ...,  1.,  0.,  0.],
            [ 0.,  0., 13., ...,  2.,  1.,  0.],
            [ 0.,  0., 16., ..., 16.,  5.,  0.],
            ...,
            [ 0.,  0., 16., ..., 15.,  0.,  0.],
            [ 0.,  0., 15., ..., 16.,  0.,  0.],
            [ 0.,  0.,  2., ...,  6.,  0.,  0.]],
    
           [[ 0.,  0.,  2., ...,  0.,  0.,  0.],
            [ 0.,  0., 14., ..., 15.,  1.,  0.],
            [ 0.,  4., 16., ..., 16.,  7.,  0.],
            ...,
            [ 0.,  0.,  0., ..., 16.,  2.,  0.],
            [ 0.,  0.,  4., ..., 16.,  2.,  0.],
            [ 0.,  0.,  5., ..., 12.,  0.,  0.]],
    
           [[ 0.,  0., 10., ...,  1.,  0.,  0.],
            [ 0.,  2., 16., ...,  1.,  0.,  0.],
            [ 0.,  0., 15., ..., 15.,  0.,  0.],
            ...,
            [ 0.,  4., 16., ..., 16.,  6.,  0.],
            [ 0.,  8., 16., ..., 16.,  8.,  0.],
            [ 0.,  1.,  8., ..., 12.,  1.,  0.]]]), 'DESCR': ".. _digits_dataset:\n\nOptical recognition of handwritten digits dataset\n--------------------------------------------------\n\n**Data Set Characteristics:**\n\n    :Number of Instances: 5620\n    :Number of Attributes: 64\n    :Attribute Information: 8x8 image of integer pixels in the range 0..16.\n    :Missing Attribute Values: None\n    :Creator: E. Alpaydin (alpaydin '@' boun.edu.tr)\n    :Date: July; 1998\n\nThis is a copy of the test set of the UCI ML hand-written digits datasets\nhttps://archive.ics.uci.edu/ml/datasets/Optical+Recognition+of+Handwritten+Digits\n\nThe data set contains images of hand-written digits: 10 classes where\neach class refers to a digit.\n\nPreprocessing programs made available by NIST were used to extract\nnormalized bitmaps of handwritten digits from a preprinted form. From a\ntotal of 43 people, 30 contributed to the training set and different 13\nto the test set. 32x32 bitmaps are divided into nonoverlapping blocks of\n4x4 and the number of on pixels are counted in each block. This generates\nan input matrix of 8x8 where each element is an integer in the range\n0..16. This reduces dimensionality and gives invariance to small\ndistortions.\n\nFor info on NIST preprocessing routines, see M. D. Garris, J. L. Blue, G.\nT. Candela, D. L. Dimmick, J. Geist, P. J. Grother, S. A. Janet, and C.\nL. Wilson, NIST Form-Based Handprint Recognition System, NISTIR 5469,\n1994.\n\n.. topic:: References\n\n  - C. Kaynak (1995) Methods of Combining Multiple Classifiers and Their\n    Applications to Handwritten Digit Recognition, MSc Thesis, Institute of\n    Graduate Studies in Science and Engineering, Bogazici University.\n  - E. Alpaydin, C. Kaynak (1998) Cascading Classifiers, Kybernetika.\n  - Ken Tang and Ponnuthurai N. Suganthan and Xi Yao and A. Kai Qin.\n    Linear dimensionalityreduction using relevance weighted LDA. School of\n    Electrical and Electronic Engineering Nanyang Technological University.\n    2005.\n  - Claudio Gentile. A New Approximate Maximal Margin Classification\n    Algorithm. NIPS. 2000."}



```python
print(digits.target)
```

    [0 1 2 ... 8 9 8]


Linear regression using SciKit-learn


```python
from sklearn.linear_model import LinearRegression
import numpy as np

#Generate training data
X = np.random.rand(100, 1)
Y = np.exp(X)

#Create linear model
linearModel = LinearRegression()
#Fit linear model to training data
linearModel.fit(X,Y)

#Generate test data
X_test = np.random.rand(1000,1)
Y_test = linearModel.predict(X_test)

plt.plot(X_test,Y_test, ".r")
plt.plot(X,Y, ".b")
plt.show()
```


![png](output_106_0.png)


#### Text analysis with TF-IDF score

Creating the document


```python
corpus = [
    'This is the first document.',
    'This is the second second document.',
    'And the third one.',
    'Is this the first document?',
]

print ("The corpus is the list: {}".format(corpus))
```

    The corpus is the list: ['This is the first document.', 'This is the second second document.', 'And the third one.', 'Is this the first document?']


Using CountVectorizer in SciKit we can implement tozenisation and occurrence counting in a single class:

Load module for sklearn:


```python
from sklearn.feature_extraction.text import CountVectorizer
```

Initiate module:


```python
vectoriser = CountVectorizer()
vectoriser
```




    CountVectorizer(analyzer='word', binary=False, decode_error='strict',
                    dtype=<class 'numpy.int64'>, encoding='utf-8', input='content',
                    lowercase=True, max_df=1.0, max_features=None, min_df=1,
                    ngram_range=(1, 1), preprocessor=None, stop_words=None,
                    strip_accents=None, token_pattern='(?u)\\b\\w\\w+\\b',
                    tokenizer=None, vocabulary=None)



Create the term freq matrix:


```python
termFreq = vectoriser.fit_transform(corpus)
vectoriser.get_feature_names()
```




    ['and', 'document', 'first', 'is', 'one', 'second', 'the', 'third', 'this']



Now we can transform the term freq output into an array:


```python
termFreq.toarray()
```




    array([[0, 1, 1, 1, 0, 0, 1, 0, 1],
           [0, 1, 0, 1, 0, 2, 1, 0, 1],
           [1, 0, 0, 0, 1, 0, 1, 1, 0],
           [0, 1, 1, 1, 0, 0, 1, 0, 1]])



To do a TF-IDF transformation, we use TfidfTransformer:


```python
from sklearn.feature_extraction.text import TfidfVectorizer
TFvector = TfidfVectorizer()
TFvector
```




    TfidfVectorizer(analyzer='word', binary=False, decode_error='strict',
                    dtype=<class 'numpy.float64'>, encoding='utf-8',
                    input='content', lowercase=True, max_df=1.0, max_features=None,
                    min_df=1, ngram_range=(1, 1), norm='l2', preprocessor=None,
                    smooth_idf=True, stop_words=None, strip_accents=None,
                    sublinear_tf=False, token_pattern='(?u)\\b\\w\\w+\\b',
                    tokenizer=None, use_idf=True, vocabulary=None)




```python
#Apply to corpus:
tfVectorisation = TFvector.fit_transform(corpus)
tfVectorisation.toarray()
```




    array([[0.        , 0.43877674, 0.54197657, 0.43877674, 0.        ,
            0.        , 0.35872874, 0.        , 0.43877674],
           [0.        , 0.27230147, 0.        , 0.27230147, 0.        ,
            0.85322574, 0.22262429, 0.        , 0.27230147],
           [0.55280532, 0.        , 0.        , 0.        , 0.55280532,
            0.        , 0.28847675, 0.55280532, 0.        ],
           [0.        , 0.43877674, 0.54197657, 0.43877674, 0.        ,
            0.        , 0.35872874, 0.        , 0.43877674]])



By default, the tf-idf vectorization returns a sparse matrix. We can see the output by converting it to a dense matrix with:


```python
print(vectoriser.vocabulary_)
tfVectorisation.todense()
```

    {'this': 8, 'is': 3, 'the': 6, 'first': 2, 'document': 1, 'second': 5, 'and': 0, 'third': 7, 'one': 4}





    matrix([[0.        , 0.43877674, 0.54197657, 0.43877674, 0.        ,
             0.        , 0.35872874, 0.        , 0.43877674],
            [0.        , 0.27230147, 0.        , 0.27230147, 0.        ,
             0.85322574, 0.22262429, 0.        , 0.27230147],
            [0.55280532, 0.        , 0.        , 0.        , 0.55280532,
             0.        , 0.28847675, 0.55280532, 0.        ],
            [0.        , 0.43877674, 0.54197657, 0.43877674, 0.        ,
             0.        , 0.35872874, 0.        , 0.43877674]])




```python
import sys
!{sys.executable} -m pip install wordcloud
```

    Collecting wordcloud
    [?25l  Downloading https://files.pythonhosted.org/packages/c7/07/e43a7094a58e602e85a09494d9b99e7b5d71ca4789852287386e21e74c33/wordcloud-1.5.0-cp37-cp37m-macosx_10_6_x86_64.whl (157kB)
    [K     |████████████████████████████████| 163kB 5.8MB/s eta 0:00:01
    [?25hRequirement already satisfied: numpy>=1.6.1 in /Users/ragyibrahim/anaconda2/envs/ML/lib/python3.7/site-packages (from wordcloud) (1.16.4)
    Collecting pillow (from wordcloud)
    [?25l  Downloading https://files.pythonhosted.org/packages/8f/f3/c6d351d7e582e4f2ef4343c9be1f0472cb249fb69695e68631e337f4b6e9/Pillow-6.1.0-cp37-cp37m-macosx_10_6_intel.macosx_10_9_intel.macosx_10_9_x86_64.macosx_10_10_intel.macosx_10_10_x86_64.whl (3.8MB)
    [K     |████████████████████████████████| 3.9MB 3.0MB/s eta 0:00:01
    [?25hInstalling collected packages: pillow, wordcloud
    Successfully installed pillow-6.1.0 wordcloud-1.5.0



```python
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt

SOME_TEXT = "A tag cloud is a visual representation for text data, typically\
used to depict keyword metadata on websites, or to visualize free form text. Some more text and tag."

wordcloud = WordCloud(stopwords = STOPWORDS, background_color= 'white', width = 1200,
                     height = 1000).generate_from_text(SOME_TEXT)

print(wordcloud.words_)
fig = plt.figure()
plt.imshow(wordcloud)
plt.show()
```

    {'text': 1.0, 'tag': 0.6666666666666666, 'cloud': 0.3333333333333333, 'visual': 0.3333333333333333, 'representation': 0.3333333333333333, 'data': 0.3333333333333333, 'typicallyused': 0.3333333333333333, 'depict': 0.3333333333333333, 'keyword': 0.3333333333333333, 'metadata': 0.3333333333333333, 'websites': 0.3333333333333333, 'visualize': 0.3333333333333333, 'free': 0.3333333333333333, 'form': 0.3333333333333333}



![png](output_124_1.png)


#### Kmeans clustering in Python

Load required mdoules


```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
```

Create sample dataset


```python
x = [1,5,1.5,8,1,9]
y=[2,8,1.8,8,0.6,11]
```

Plot data


```python
plt.scatter(x,y)
plt.show()
```


![png](output_131_0.png)


Now lets create a matrix X with x,y coordinates


```python
X = np.array([
    [1,2],
    [5,8],
    [1.5, 1.8],
    [8,8],
    [1,0.6],
    [9,11]
])
```

Initiate K-Means algorithm with 2 clusters


```python
kmeans_1 = KMeans(n_clusters=2)
kmeans_1.fit(X)
```




    KMeans(algorithm='auto', copy_x=True, init='k-means++', max_iter=300,
           n_clusters=2, n_init=10, n_jobs=None, precompute_distances='auto',
           random_state=None, tol=0.0001, verbose=0)



Now, we have fit the KMeans model to our data, X. The model will have identified 2 clusters, with 2 cluster centres (centroids). we can get this data as:


```python
centroids = kmeans_1.cluster_centers_
labels = kmeans_1.labels_

print(centroids)
print(labels)
```

    [[1.16666667 1.46666667]
     [7.33333333 9.        ]]
    [0 1 0 1 0 1]


Lets try to visualise the clusters by plotting them. The centroids will be marked as “X”


```python
colors = ['g.', 'r.', 'c.', 'y.']

for i in range(len(X)):
    print("coordinates: ", X[i], "label: ", labels[i])
    plt.plot(X[i][0], X[i][1], colors[labels[i]], markersize = 10)
    
#Visualise the centroids
plt.scatter(centroids[:, 0], centroids[:, 1], marker = "X", s = 150, linewidths = 1, zorder = 10 )
plt.show()
```

    coordinates:  [1. 2.] label:  0
    coordinates:  [5. 8.] label:  1
    coordinates:  [1.5 1.8] label:  0
    coordinates:  [8. 8.] label:  1
    coordinates:  [1.  0.6] label:  0
    coordinates:  [ 9. 11.] label:  1



![png](output_139_1.png)



```python
import numpy as np
```


```python
x=np.array([[1,0],[0,-1]])
y=np.array([[-3,-2], [-4,1], [0,4], [4,1], [2,-3]])


z= np.dot(y,x)
print(z)
```

    [[-3  2]
     [-4 -1]
     [ 0 -4]
     [ 4 -1]
     [ 2  3]]



```python

```
