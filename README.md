# DeepC

**DeepC** is a lightweight neural network library written in C developed as part of [Introduction to Deep Learning (CSE 599g1)](https://courses.cs.washington.edu/courses/cse599g1/18au/) course in Fall 2018. The ```src/``` subfolder contains the header and source files. ```uwnet.py``` is a wrapper function which provides a pure Python interface to our C library. ```trycifar.py``` and ```trymnist.py``` are python scripts where we load the data, define our network and run the train and test functions.

## 1. matrix

```matrix``` is the most fundamental data structure in this framework. It contains 4 variables, the number of rows and columns of the matrix, an array of floats containing data in row-major order. The ```matrix``` struct is declared in the ```matrix.h``` file. The file also declares functions to manipulate and copy matrices which are later defined in ```matrix.c```. 

## 2. data

```data``` struct is declared and defined in ```uwnet.h```. It contains the input matrix **X** and the output matrix **y** on which the neural network is trained.

## 3. activation functions

```activations.c``` defines ```activate_matrix``` and ```gradient_matrix``` for different activation function. Right now, it supports Logistic, RELU, LRELU and Softmax activation function.

### 3.1 ```activate_matrix```

It takes a matrix ```m``` and an activation function ```a```  which is a constant string and returns ```f(m)```, where ```f``` is defined by what the activation ```a``` is. ```f``` is applied elementwise to ```m```.

### 3.2 ```gradient_matrix```

Given the output of a layer i.e ```f(m)``` and the activation function ```a``` that defines ```f```, it returns ```f'(m)```.

## 4. layer

The ```layer``` struct contains the input and output of a layer. It also stores the weights and biases. It  

### 4.1 connected_layer 

It is a fully connected layer defined in ```connected_layer.c```. It has three functions ```forward_connected_layer```, ```backward_connected_layer``` and ```update_connected_layer```.

#### 4.1.1 forwad_connected_layer

The function outputs a matrix ```f(in*w + b)``` where: ```in``` is the input, ```w``` is the weights, ```b``` is the bias, and ```f``` is the activation function.

#### 4.1.2 backward_connected_layer

The function takes ```dL/df(in*w+b)``` (derivative of loss wrt output) as the argument and returns ```dL/dw``` (derivative of 
loss wrt layer weights), ```dL/db```(derivative of loss wrt biases) and ```dL/din``` (derivative of loss wrt to input).

#### 4.1.3 update_connected_layer

The function updates our weights using SGD with momentum and weight decay.

### 4.2 convolutional_layer

```convolutional_layer.c``` defines the convolutional layer. It contains the following major functions: ```im2col```, ```col2im```, ```forward_convolutional_layer```, ```backward_convolutional_layer``` and ```update_convolutional_layer```

#### 4.2.1 im2col

It take spatial blocks of image and put them into columns of a matrix. ```Im2col``` handles kernel size, stride,
padding, etc.

![im2col example](images/im2col.gif)

### 4.2.2 col2im

#### 4.2.3 forwad_convolutional_layer

#### 4.2.4 backward_convolutional_layer

#### 4.2.5 update_convolutional_layer

### 4.3 maxpool_layer

#### 4.3.1 forwad_maxpool_layer

#### 4.3.2 backward_maxpool_layer

#### 4.3.3 update_maxpool_layer


## 4. net

### 4.1 forward_net

### 4.2 backward_net

### 4.3 update_net

## 5. classifier

```classifier``` is a wrapper around ```net.c```.

The ```net``` struct is an array of layers. It also stores a variale which is equal to the depth of the net

## Compile and Run

Run ```make``` to compile your source file and to make sure your executables and libraries are up to date. Then either run ```python trymnist.py``` or  ```python trycifar.py``` to train the  neural network on MNIST and CIFAR-10 data respectivey. All data resides in individual subfolders in the ```./data/``` folder.
