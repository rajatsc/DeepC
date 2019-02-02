#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <assert.h>
#include <time.h>
#include <sys/time.h>
#include "matrix.h"
#include "image.h"
#include "test.h"
#include "args.h"
#include "uwnet.h"

double what_time_is_it_now()
{
    struct timeval time;
    if (gettimeofday(&time,NULL)){
        return 0;
    }
    return (double)time.tv_sec + (double)time.tv_usec * .000001;
}

void test_matrix_speed()
{
    int i;
    int n = 128;
    matrix a = random_matrix(512, 512, 1);
    matrix b = random_matrix(512, 512, 1);
    double start = what_time_is_it_now();
    for(i = 0; i < n; ++i){
        matrix d = matmul(a,b);
        free_matrix(d);
    }
    printf("Matmul elapsed %lf sec\n", what_time_is_it_now() - start);
    start = what_time_is_it_now();
    for(i = 0; i < n; ++i){
        matrix at = transpose_matrix(a);
        free_matrix(at);
    }
    printf("Transpose elapsed %lf sec\n", what_time_is_it_now() - start);
}



void test_im2col()
{
    float data[24] = {0};
    float data2[24] = {0};
    int w=4;
    int h=3;
    int c=2;

    int i;
    for (i=1; i <=24; i++){
        float x= (float) i;
        data[i-1]=x;
        printf("%f\n", data[i-1]);
    }


    image myimage=float_to_image(data, w, h, c);

    printf("Width:%d \n", myimage.w);
    printf("Height:%d \n", myimage.h);
    printf("Channel:%d \n", myimage.c);

    
    int size=2;
    int stride=2;
    matrix mymatrix=im2col(myimage, size, stride);

    print_matrix(mymatrix);
    
    image myimage2=float_to_image(data2, w, h, c);
    printf("called col2im inside test \n");
    col2im(mymatrix, size, stride, myimage2);

    matrix mymatrix2=im2col(myimage2, size, stride);
    print_matrix(mymatrix2);
}


void test_forward_maxpool_layer()
{
	int w=3;
	int h=2;
	int c=2;
	int size=3;
	int stride=2;
	int tot_images=5;
	layer mylayer=make_maxpool_layer(w, h, c, size, stride);

	matrix in=random_matrix(tot_images, w*h*c, 1);

	matrix out= forward_maxpool_layer(mylayer, in);

	print_matrix(in);
	print_matrix(out);
}


void test_backward_maxpool_layer()
{
    
    int w=3;
    int h=2;
    int c=2;
    int size=3;
    int stride=2;
    int tot_images=5;
    layer mylayer=make_maxpool_layer(w, h, c, size, stride);

    int outw = (mylayer.width-1)/mylayer.stride + 1;
    int outh = (mylayer.height-1)/mylayer.stride + 1;

    matrix prev_delta=make_matrix(tot_images, w*h*c);
    matrix in=random_matrix(tot_images, w*h*c, 2.0);


    matrix delta=random_matrix(tot_images, outw*outh*c, 2.0);

    mylayer.delta[0]=copy_matrix(delta);
    mylayer.in[0]=copy_matrix(in);

    printf("Printing before backward maxpool \n");
    printf("Input \n");
    print_matrix(mylayer.in[0]);
    printf("\n");
    printf("prev_delta \n");
    print_matrix(prev_delta);
    printf("\n");
    printf("delta \n");
    print_matrix(mylayer.delta[0]);
    printf("\n");
    backward_maxpool_layer(mylayer, prev_delta);
    printf("Printing after maxpool operation \n");
    printf("-----------------------------------\n");
    printf("prev_delta \n");
    print_matrix(prev_delta);
}

void random_tests()
{
    printf("just a random test \n");
}   


void run_tests()
{
    test_backward_maxpool_layer();
    //test_forward_maxpool_layer();
    //test_im2col();
    //test_matrix_speed();
    //printf("%d tests, %d passed, %d failed\n", tests_total, tests_total-tests_fail, tests_fail);
}

