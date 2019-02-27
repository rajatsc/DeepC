# DeepC

**DeepC** is a lightweight neural network framework written in C developed as part of [Introduction to Deep Learning (CSE 599g1)](https://courses.cs.washington.edu/courses/cse599g1/18au/) course in Fall 2018.

## matrix

```matrix``` is the most fundamental data structure in this framework. It contains 4 variables, the number of rows and columns of the matrix, and data stored as an array of floats in row-major order. The ```matrix``` struct is declared in the ```matrix.h``` file. The file also declares functions to manipulate and copy matrices which are defined in ```matrix.c```. 

## data


## layer


## net


Run ```make``` to compile your source file and to make sure your executables and libraries are up to date. Then either run ```python trymnist.py``` or  ```python trycifar.py``` to train the  neural network on MNIST and CIFAR-10 data respectivey. All data resides in individual subfolders in the ```./data/``` folder.