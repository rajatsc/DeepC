#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include "uwnet.h"
#include "matrix.h"
#include <stdio.h>

// Add bias terms to a matrix
// matrix m: partially computed output of layer
// matrix b: bias to add in (should only be one row!)
void forward_bias(matrix m, matrix b)
{
    assert(b.rows == 1);
    assert(m.cols == b.cols);
    int i,j;
    for(i = 0; i < m.rows; ++i){
        for(j = 0; j < m.cols; ++j){
            m.data[i*m.cols + j] += b.data[j];
        }
    }
}

// Calculate bias updates from a delta matrix
// matrix delta: error made by the layer
// matrix db: delta for the biases
void backward_bias(matrix delta, matrix db)
{
    int i, j;
    for(i = 0; i < delta.rows; ++i){
        for(j = 0; j < delta.cols; ++j){
            db.data[j] += delta.data[i*delta.cols + j];
        }
    }
}

// Run a connected layer on input
// layer l: pointer to layer to run
// matrix in: input to layer
// returns: the result of running the layer: f(wx + b)
matrix forward_connected_layer(layer l, matrix in)
{   
    
    //printf("inside forward connected layer \n");
    // TODO: 3.1 - run the network forward
    //matrix out = make_matrix(in.rows, l.w.cols); // Going to want to change this!
    matrix out = matmul(in, l.w);
    
    //matrix u = copy_matrix(out);
    forward_bias(out, l.b);
    activate_matrix(out, l.activation);

    // Saving our input and output and making a new delta matrix to hold errors
    // Probably don't change this
    l.in[0] = in;
    free_matrix(l.out[0]);
    l.out[0] = out;
    free_matrix(l.delta[0]);
    l.delta[0] = make_matrix(out.rows, out.cols);
    
    //free_matrix(l.u[0]);
    //l.u[0] = u;
    //printf("forward connectd layer completed \n");
    return out;
}

// Run a connected layer backward
// layer l: layer to run
// matrix delta: 
void backward_connected_layer(layer l, matrix prev_delta)
{   
    //printf("inside backward connected layer \n");
    matrix in    = l.in[0];
    matrix out   = l.out[0];
    matrix delta = l.delta[0];
    //matrix u     = l.u[0];

    // TODO: 3.2
    // delta is the error made by this layer, dL/dout
    // First modify in place to be dL/d(in*w+b) using the gradient of activation
    
    // Calculate the updates for the bias terms using backward_bias
    // The current bias deltas are stored in l.db

    // Then calculate dL/dw. Use axpy to add this dL/dw into any previously stored
    // updates for our weights, which are stored in l.dw
    
    
    gradient_matrix(out, l.activation, delta);
    //print_matrix(delta);
    backward_bias(delta, l.db);
    matrix in_tr=transpose_matrix(in);
    matrix del_w;

    del_w=matmul(in_tr, delta);
    axpy_matrix(1, del_w, l.dw);

    // Then calculate dL/dw. Use axpy to add this dL/dw into any previously stored
    // updates for our weights, which are stored in l.dw

    int i;
    if(prev_delta.data){
        //print_matrix(prev_delta);
        matrix w_tr=transpose_matrix(l.w);
        matrix tmp_mat=matmul(delta, w_tr);
        //prev_delta=matmul(delta, w_tr);
        //print_matrix(tmp_mat);
        //prev_delta=copy_matrix(tmp_mat);
        //print_matrix(prev_delta);
        /*
        for (i=0; i < tmp_mat.rows*tmp_mat.cols; i++){
            prev_delta.data[i]=tmp_mat.data[i];    
        }
        */
        axpy_matrix(1, tmp_mat, prev_delta);
        
        free_matrix(w_tr);
    	free_matrix(tmp_mat);
    }

    /*
    free_matrix(in);
    free_matrix(out);
    free_matrix(delta);
    free_matrix(in_tr);
    free_matrix(del_w);
	*/

    //printf("backward connected layer completed \n");
}

// Update 
void update_connected_layer(layer l, float rate, float momentum, float decay)
{    
    //printf("in update conected layer \n");
    /*
    //-----------updating weights------------------------
    axpy_matrix(-decay, l.w, l.dw);
    axpy_matrix(rate, l.dw, l.w); 
    scal_matrix(momentum, l.dw);

    
    //------------updating biases------------------------
    axpy_matrix(-decay, l.b, l.db);
    axpy_matrix(rate, l.db, l.b);
    scal_matrix(momentum, l.db);
    //printf("update connected layer completed \n");
    */

    axpy_matrix(rate, l.db, l.b);
    scal_matrix(momentum, l.db);

    axpy_matrix(-decay, l.w, l.dw);
    axpy_matrix(rate, l.dw, l.w);
    scal_matrix(momentum, l.dw);


}

layer make_connected_layer(int inputs, int outputs, ACTIVATION activation)
{
    layer l = {0};
    l.w  = random_matrix(inputs, outputs, sqrtf(2.f/inputs));
    l.dw = make_matrix(inputs, outputs);
    l.b  = make_matrix(1, outputs);
    l.db = make_matrix(1, outputs);
    l.in = calloc(1, sizeof(matrix));
    l.out = calloc(1, sizeof(matrix));
    l.delta = calloc(1, sizeof(matrix));

    //batch-norm parameters
    l.x = calloc(1, sizeof(matrix));
    l.rolling_mean = make_matrix(1, outputs);
    l.rolling_variance = make_matrix(1, outputs);
    //l.u = calloc(1, sizeof(matrix));
    l.activation = activation;
    l.forward = forward_connected_layer;
    l.backward = backward_connected_layer;
    l.update=update_connected_layer;
    return l;
}

