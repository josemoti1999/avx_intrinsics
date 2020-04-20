#include <iostream>
#include <stdlib.h>
#include "util2.h"
#include <time.h>

#define N 1
#define C 3
#define H 227
#define W 227

int main()
{
  AlexNet net;
  
  fmap input;
  input.dim1 = N;
  input.dim2 = C;
  input.dim3 = H;
  input.dim4 = W;
  input.data = (DATA*) malloc(N * C * H * W * sizeof(DATA));

  DATA (*temp)[H][W][C] = (DATA (*)[H][W][C])input.data;

  for(int i=0; i<N; i++)
    for(int j=0; j<C; j++)
      for(int k=0; k<H; k++)
        for(int l=0; l<W; l++)
          temp[i][j][k][l] = (i*C*H*W+j*C*W+k*C+l)%256;    //os2 ws2 is2

  fmap* output = net.forward_pass(&input);

  for(int i=0; i<5; i++)
    std::cout<<"conv"<<i<<"  "<<net.conv_layers[i]->exec_time<<std::endl;
  
  for(int i=0; i<3; i++)
    std::cout<<"fc"<<i<<"  "<<net.linear_layers[i]->exec_time<<std::endl;

  std::cout<<"Total time  "<<net.exec_time<<std::endl;
  return 0;
}
