# DeepC

**DeepC** is a lightweight neural network library written in C developed as part of [Introduction to Deep Learning (CSE 599g1)](https://courses.cs.washington.edu/courses/cse599g1/18au/) course in Fall 2018. The ```./src/``` subfolder contains the header and source files. ```uwnet.py``` is a wrapper function which provides a pure Python interface to our C library. ```trycifar.py``` and ```trymnist.py``` are python scripts where we load the data, define our network and run the training and testing functions.

## 1. matrix

```matrix``` is the most fundamental data structure in this framework. It contains 4 variables, the number of rows and columns of the matrix, an array of floats containing data in row-major order. The ```matrix``` struct is declared in the ```matrix.h``` file. The file also declares functions to manipulate and copy matrices which are later defined in ```matrix.c```. 

## 2. data

```data``` struct is declared and defined in ```uwnet.h```. It contains the input matrix **X** and the output matrix **y** on which the neural network is trained.


## 3. layer

The ```layer``` struct contains the input and input to a layer. It also stores the weights and biases 

### 3.1 connected_layer 

Our layer outputs a matrix called out as ```f(in*w + b)``` where: ```in``` is the input, ```w``` is the weights, ```b``` is the bias, and ```f``` is the activation function. ```activate_matrix``` and ```gradient_matrix``` functions are defined in ```activations.c``` file 


Finally, we'll want to activate the output with the activation function for that layer.

#### 3.1.1 forwad_connected_layer

#### 3.1.2 backward_connected_layer

#### 3.1.3 update_connected_layer

### 3.2 convolutional_layer

#### 3.2.1 forwad_convolutional_layer

#### 3.2.2 backward_convolutional_layer

#### 3.2.3 update_convolutional_layer

### 3.3 maxpool_layer

#### 3.3.1 forwad_maxpool_layer

#### 3.3.2 backward_maxpool_layer

#### 3.3.3 update_maxpool_layer


## 4. net

### 4.1 forward_net

### 4.2 backward_net

### 4.3 update_net

## 5. classifier

```classifier``` is a wrapper around ```net.c```.

The ```net``` struct is an array of layers. It also stores a variale which is equal to the depth of the net

## Compile and Run

Run ```make``` to compile your source file and to make sure your executables and libraries are up to date. Then either run ```python trymnist.py``` or  ```python trycifar.py``` to train the  neural network on MNIST and CIFAR-10 data respectivey. All data resides in individual subfolders in the ```./data/``` folder.