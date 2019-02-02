from uwnet import *

def conv_net():
    l = [   make_convolutional_layer(32, 32, 3, 8, 3, 1, LRELU, batchnorm=1),
            make_maxpool_layer(32, 32, 8, 3, 2),
            make_convolutional_layer(16, 16, 8, 16, 3, 1, LRELU, batchnorm=1),
            make_maxpool_layer(16, 16, 16, 3, 2),
            make_convolutional_layer(8, 8, 16, 32, 3, 1, LRELU, batchnorm=1),
            make_maxpool_layer(8, 8, 32, 3, 2),
            make_convolutional_layer(4, 4, 32, 64, 3, 1, LRELU, batchnorm=1),
            make_maxpool_layer(4, 4, 64, 3, 2),
            make_connected_layer(256, 10, SOFTMAX)]
    return make_net(l)

print("loading data...")
train = load_image_classification_data("cifar/cifar.train", "cifar/cifar.labels")
test  = load_image_classification_data("cifar/cifar.test",  "cifar/cifar.labels")
print("done")
print

print("making model...")
batch = 512
iters = 2500
rate = [0.1, 0.05, 0.01, 0.005, 0.001]
momentum = .9
decay = .005

m = conv_net()
print("training...")
#train_image_classifier(m, train, batch, int(iters*0.05), rate[0], momentum, decay)
train_image_classifier(m, train, batch, int(iters/5), rate[0], momentum, decay)
train_image_classifier(m, train, batch, int(iters/5), rate[1], momentum, decay)
train_image_classifier(m, train, batch, int(iters/5), rate[2], momentum, decay)
train_image_classifier(m, train, batch, int(iters/5), rate[3], momentum, decay)
train_image_classifier(m, train, batch, int(iters/5), rate[4], momentum, decay)
#train_image_classifier(m, train, batch, int(iters*0.50), rate[1], momentum, decay)

print("done")
print

print("evaluating model...")
print("training accuracy: %f", accuracy_net(m, train))
print("test accuracy:     %f", accuracy_net(m, test))

###############################################################################
##############################################################################

"""

You should be able to add it to convolutional or connected layers. 
The standard for batch norm is to use it at every layer except the output.
First, train the conv_net as usual. Then try it with batchnorm. Does it
do better??

In class we learned about annealing your learning rate to get better
convergence. We ALSO learned that with batch normalization you
can use larger learning rates because it's more stable. Increase
the starting learning rate to .1 and train for multiple rounds with
successively smaller learning rates. Using just this model, what's
the best performance you can get?


"""

