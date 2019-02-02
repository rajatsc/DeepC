#include <math.h>
#include <string.h>
#include <stdlib.h>
#include "uwnet.h"
#include "image.h"
#include "test.h"
#include "args.h"

void try_mnist()
{
    data train = load_image_classification_data("mnist/mnist.train", "mnist/mnist.labels");
    data test  = load_image_classification_data("mnist/mnist.test", "mnist/mnist.labels");

    net n = {0};
    n.layers = calloc(3, sizeof(layer));
    n.n = 3;
    //n.layers[0] = make_connected_layer(784, 32, LRELU);
    n.layers[0] = make_convolutional_layer(28, 28, 1, 1, 5, 2, LRELU);
    n.layers[1] = make_convolutional_layer(14, 14, 1, 8, 5, 2, LRELU);
    n.layers[2] = make_connected_layer(392, 10, SOFTMAX);

    int batch = 128;
    int iters = 5000;
    float rate = .01;
    float momentum = .9;
    float decay = .0005;

    train_image_classifier(n, train, batch, iters, rate, momentum, decay);
    printf("Training accuracy: %f\n", accuracy_net(n, train));
    printf("Testing  accuracy: %f\n", accuracy_net(n, test));
}

int main(int argc, char **argv)
{
    if(argc < 2){
        printf("usage: %s [test | trymnist]\n", argv[0]);  
    } else if (0 == strcmp(argv[1], "trymnist")){
        try_mnist();
    } else if (0 == strcmp(argv[1], "test")){
        run_tests();
    }
    return 0;
}
