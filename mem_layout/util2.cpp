#include <stdlib.h>
#include <iostream>
#include "util2.h"
#include <time.h>
#include <immintrin.h>

fmap create_fmap(int N, int C, int H, int W){
    //for creating a new object of type fmap
    fmap input;
    input.dim1 = N;
    input.dim2 = C;
    input.dim3 = H;
    input.dim4 = W;
    input.data = (DATA*) malloc(N * C * H * W * sizeof(DATA));
    return input;
}

Convolution::Convolution(int m, int c, int r, int s, int sx, int sy, int px, int py)
{
  M = m;
  C = c;
  R = r;
  S = s;
  Sx = sx;
  Sy = sy;
  Px = px;
  Py = py;
  weights = (DATA*) malloc(M * C * R * S * sizeof(DATA));
  DATA (*temp_weights)[R][S][C] = (DATA (*)[R][S][C])weights;
  for(int i=0; i<M; i++)
    for(int j=0; j<R; j++)
      for(int k=0; k<S; k++)
        for(int l=0; l<C; l++)
          temp_weights[i][j][k][l] = (i*C*R*S+j*C*S+k*C+l)%256;    //os2,is2,ws2
}

Linear::Linear(int m, int l)
{
  //L-C*H*W, M-number of neurons in output, N-batch size
  M = m;
  L = l;
  weights = (DATA*) malloc(M * L * sizeof(DATA));
  DATA (*temp)[L] = (DATA (*)[L])weights;
  for(int i=0; i<M; i++)
    for(int j=0; j<L; j++)
      temp[i][j] = (i*L+j)%256;
}

fmap* Convolution::conv_2d(fmap* input_features)
{
    //does convolution operation without any avx intrinsics
    
    int start,end;
    start=clock();
    
    //declare all parameters
    int N=input_features->dim1;
    int C=input_features->dim2;
    int W=input_features->dim3;
    int H=input_features->dim4;
    int E = int((H + 2*Py-S)/Sy + 1);
    int F = int((W + 2*Px-R)/Sx + 1);
    
    //padding, H_ and W_ are the new dimensions
    int H_ = int(H + Py*2);
    int W_ = int(W + Px*2);
    fmap* input_new = (fmap*) malloc(sizeof(create_fmap(N, C , H_, W_)));
    *input_new = create_fmap(N, C, H_, W_);
    DATA (*temp_inputs_new)[C][H_][W_] = (DATA (*)[C][H_][W_])input_new->data;
    DATA (*temp_inputs)[C][H][W] = (DATA (*)[C][H][W])input_features->data;
    for(int n=0; n<N; n++){
        for(int c=0;c<C;c++){
            for(int y=0;y<H_;y++){
                for(int x=0;x<W_;x++){
                    if((x>=Px && x<F-Px) && (y>=Py && y<E-Py)){
                        temp_inputs_new[n][c][y][x] = 
                            temp_inputs[n][c][y-Py][x-Px];
                    }
                    else{
                        temp_inputs_new[n][c][y][x] = 0;
                    }
                }
            }
        }
    }
    free(input_features); //free memory allocated for older input features
    //std::cout<<"padding done"<<std::endl;
    
    //(1.)output feature map allocation
    fmap* output_features = (fmap*) malloc(sizeof(create_fmap(N, M, E, F)));
    *output_features = create_fmap(N, M, E, F);
    DATA (*temp_outputs)[M][E][F] = (DATA (*)[M][E][F])output_features->data;
    DATA (*temp_weights)[C][R][S] = (DATA (*)[C][R][S])weights;    

    //(2.)computation with for loop
    for(int n=0; n<N; n++){
        for(int m=0; m<M; m++){
            for(int x=0; x<F; x+=1){
                for(int y=0; y<E; y+=1){
                    temp_outputs[n][m][y][x] = 0;
                    for(int c=0; c<C; c++){
                        for(int j=0; j<S; j++){
                            for(int i=0; i<R; i++){
                                temp_outputs[n][m][y][x]+= 
                                    temp_inputs_new[n][c][Sy*y+j][Sx*x+i]
                                    *temp_weights[m][c][j][i];
                            }
                        }
                    }
                }
            }
        }
    }
    
    //(3.)free input feature memory
    free(input_new);
    end =clock();
    exec_time = double(end-start) / double(CLOCKS_PER_SEC);
    //std::cout<<"conv done"<<std::endl;
    
    //(4.)return output
    return output_features;
}

fmap* Convolution::conv2d_IS(fmap* input_features)
{
    //input stationary using avx        IS2
    
    int start,end;
    start=clock();
    
    //declare all parameters
    int N=input_features->dim1;
    int C=input_features->dim2;
    int W=input_features->dim3;
    int H=input_features->dim4;
    int E = int((H + 2*Py-S)/Sy + 1);
    int F = int((W + 2*Px-R)/Sx + 1);
    
    //padding, H_ and W_ are the new dimensions
    int H_ = int(H + Py*2);
    int W_ = int(W + Px*2);
    fmap* input_new = (fmap*) malloc(sizeof(create_fmap(N, C , H_, W_)));
    *input_new = create_fmap(N, C, H_, W_);
    DATA (*temp_inputs_new)[H_][W_][C] = (DATA (*)[H_][W_][C])input_new->data;
    DATA (*temp_inputs)[H][W][C] = (DATA (*)[H][W][C])input_features->data;
    for(int n=0; n<N; n++){
        for(int c=0;c<C;c++){
            for(int y=0;y<H_;y++){
                for(int x=0;x<W_;x++){
                    if((x>=Px && x<F-Px) && (y>=Py && y<E-Py)){
                        temp_inputs_new[n][y][x][c] = 
                            temp_inputs[n][y-Py][x-Px][c];
                    }
                    else{
                        temp_inputs_new[n][y][x][c] = 0;
                    }
                }
            }
        }
    }
    free(input_features); //free memory allocated for older input features
    //std::cout<<"padding done"<<std::endl;
    
    //(1.)output feature map allocation
    fmap* output_features = (fmap*) malloc(sizeof(create_fmap(N, M, E, F)));
    *output_features = create_fmap(N, M, E, F);
    DATA (*temp_outputs)[M][E][F] = (DATA (*)[M][E][F])output_features->data;
    DATA (*temp_weights)[R][S][C] = (DATA (*)[R][S][C])weights;    

    //(2.)computation with for loop input stationary
    __m256 mm_inputs, mm_weights, product,sum;
    __m256i mask;
    float array[8];    
    for(int n=0; n<N; n++){
         for(int e=0; e<E;e+=1){
             for(int f=0; f<F; f+=1){     //N,C,H_,W_ are output parameters
                 for(int s=0; s<S; s++){                 
                    for(int r=0; r<R; r++){
                        for(int c=0; c<C/8; c++){
                            int index = C-c*8-8;
                            mask=_mm256_set_epi32(index+8,index+7, index+6, index+5, 
                                                      index+4,index+3, index+2,index+1);
                            mm_inputs=_mm256_maskload_ps((float const*) &temp_inputs_new[n][Sy*e+r][Sx*f+s][c*8], mask);
                            for(int m=0; m<M; m++){
                            sum=_mm256_setzero_ps();
                            for(int r=0; r<R; r++){
                            for(int s=0; s<S/8; s++){ 
                                int index = S-s*8-8;
                                mask=_mm256_set_epi32(index+8,index+7, index+6, index+5, 
                                                      index+4,index+3, index+2,index+1);
                                mm_weights=_mm256_maskload_ps((float const*) &temp_weights[m][r][s][c*8], mask);
                                product=_mm256_mul_ps(mm_inputs, mm_weights);
                                sum=_mm256_add_ps(sum, product);
                            }
                            }
                            _mm256_storeu_ps((float*) &array,sum);
                            for(int qa=0;qa<8;qa++){
                                temp_outputs[n][m][e][f]+=array[qa];
                            }
                            }
                        }
                    }
                 }
             }
         }
    }                                
                          
    //(3.)free input feature memory
    free(input_new);
    end =clock();
    exec_time = double(end-start) / double(CLOCKS_PER_SEC);
    //std::cout<<"conv done"<<std::endl;
    
    //(4.)return output
    return output_features;
}

fmap* Convolution::conv2d_OS(fmap* input_features)
{
    //output stationary using avx           OS2
    
    int start,end;
    start=clock();
    
    //declare all parameters
    int N=input_features->dim1;
    int C=input_features->dim2;
    int W=input_features->dim3;
    int H=input_features->dim4;
    int E = int((H + 2*Py-S)/Sy + 1);
    int F = int((W + 2*Px-R)/Sx + 1);
    
    //padding, H_ and W_ are the new dimensions
    int H_ = int(H + Py*2);
    int W_ = int(W + Px*2);
    fmap* input_new = (fmap*) malloc(sizeof(create_fmap(N, C , H_, W_)));
    *input_new = create_fmap(N, C, H_, W_);
    DATA (*temp_inputs_new)[H_][W_][C] = (DATA (*)[H_][W_][C])input_new->data;
    DATA (*temp_inputs)[H][W][C] = (DATA (*)[H][W][C])input_features->data;
    for(int n=0; n<N; n++){
        for(int c=0;c<C;c++){
            for(int y=0;y<H_;y++){
                for(int x=0;x<W_;x++){
                    if((x>=Px && x<F-Px) && (y>=Py && y<E-Py)){
                        temp_inputs_new[n][y][x][c] = 
                            temp_inputs[n][y-Py][x-Px][c];
                    }
                    else{
                        temp_inputs_new[n][y][x][c] = 0;
                    }
                }
            }
        }
    }
    free(input_features); //free memory allocated for older input features
    //std::cout<<"padding done"<<std::endl;
    
    //(1.)output feature map allocation
    fmap* output_features = (fmap*) malloc(sizeof(create_fmap(N, M, E, F)));
    *output_features = create_fmap(N, M, E, F);
    DATA (*temp_outputs)[M][E][F] = (DATA (*)[M][E][F])output_features->data;
    DATA (*temp_weights)[R][S][C] = (DATA (*)[R][S][C])weights;    

    //(2.)computation with for loop output stationary
    __m256 mm_inputs, mm_weights, product,sum;
    __m256i mask;
    float array[8];
    for(int n=0; n<N; n++){
        for(int m=0; m<M; m++){
            for(int e=0; e<E;e+=1){
                for(int f=0; f<F; f+=1){                 //N,M,E,F are output parameters
                    sum=_mm256_setzero_ps();
                    for(int s=0; s<S; s++){ 
                        for(int r=0; r<R; r++){
                            for(int c=0; c<C/8; c++){                              
                                int index = C-c*8-8;
                                mask=_mm256_set_epi32(index+8,index+7, index+6, index+5, 
                                                      index+4,index+3, index+2,index+1);
                                mm_inputs=_mm256_maskload_ps((float const*) &temp_inputs_new[n][Sy*e+r][Sx*f+s][c*8], mask);
                                mm_weights=_mm256_maskload_ps((float const*) &temp_weights[m][r][s][c*8], mask);
                                product=_mm256_mul_ps(mm_inputs, mm_weights);
                                sum=_mm256_add_ps(sum, product);
                            }
                        }
                    }
                    _mm256_storeu_ps((float*) &array,sum);
                    temp_outputs[n][m][e][f]=0;
                    for(int qa=0;qa<8;qa++){
                        temp_outputs[n][m][e][f]+=array[qa];
                    }                    
                }
            }
        }
    }
                                  
    //(3.)free input feature memory
    free(input_new);
    end =clock();
    exec_time = double(end-start) / double(CLOCKS_PER_SEC);
    //std::cout<<"conv done"<<std::endl;
    
    //(4.)return output
    return output_features;
}


fmap* Convolution::conv2d_WS(fmap* input_features)
{
    //weight stationary using avx             WS1
    
    int start,end;
    start=clock();
    
    //declare all parameters
    int N=input_features->dim1;
    int C=input_features->dim2;
    int W=input_features->dim3;
    int H=input_features->dim4;
    int E = int((H + 2*Py-S)/Sy + 1);
    int F = int((W + 2*Px-R)/Sx + 1);
    
    //padding, H_ and W_ are the new dimensions
    int H_ = int(H + Py*2);
    int W_ = int(W + Px*2);
    fmap* input_new = (fmap*) malloc(sizeof(create_fmap(N, C , H_, W_)));
    *input_new = create_fmap(N, C, H_, W_);
    DATA (*temp_inputs_new)[C][H_][W_] = (DATA (*)[C][H_][W_])input_new->data;
    DATA (*temp_inputs)[C][H][W] = (DATA (*)[C][H][W])input_features->data;
    for(int n=0; n<N; n++){
        for(int c=0;c<C;c++){
            for(int y=0;y<H_;y++){
                for(int x=0;x<W_;x++){
                    if((x>=Px && x<F-Px) && (y>=Py && y<E-Py)){
                        temp_inputs_new[n][c][y][x] = 
                            temp_inputs[n][c][y-Py][x-Px];
                    }
                    else{
                        temp_inputs_new[n][c][y][x] = 0;
                    }
                }
            }
        }
    }
    free(input_features); //free memory allocated for older input features
    //std::cout<<"padding done"<<std::endl;
    
    //(1.)output feature map allocation
    fmap* output_features = (fmap*) malloc(sizeof(create_fmap(N, M, E, F)));
    *output_features = create_fmap(N, M, E, F);
    DATA (*temp_outputs)[M][E][F] = (DATA (*)[M][E][F])output_features->data;
    DATA (*temp_weights)[C][R][S] = (DATA (*)[C][R][S])weights;    

    //(2.)computation with for loop weight stationary 
    __m256 mm_inputs, mm_weights, product;
    __m256i mask;
    float array[8];
    for(int m=0; m<M; m++){
        for(int c=0; c<C; c++){
            for(int r=0; r<R; r++){
                for(int s=0; s<S/8; s++){                //M,C,R,S are weight dimensions
                    int index = S-s*8-8;
                    mask=_mm256_set_epi32(index+8,index+7, index+6, index+5, index+4,index+3, index+2,index+1);
                    mm_weights=_mm256_maskload_ps((float const*) &temp_weights[m][c][r][s*8], mask);
                    for(int n=0; n<N; n++){
                        for(int e=0; e<E; e+=1){
                            for(int f=0; f<F;f+=1){
                                mm_inputs=_mm256_maskload_ps((float const*) &temp_inputs_new[n][c][Sy*e+r][Sx*f+s*8], mask);
                                product=_mm256_mul_ps(mm_inputs, mm_weights);
                                _mm256_storeu_ps((float*) &array,product);
                                //initialization problem
                                for(int qa=0;qa<8;qa++){
                                    temp_outputs[n][m][e][f]+=array[qa];                                    
                                }
                            }
                        }
                    }
                }
            }
        }
    }
                                  
    //(3.)free input feature memory
    free(input_new);
    end =clock();
    exec_time = double(end-start) / double(CLOCKS_PER_SEC);
    //std::cout<<"conv done"<<std::endl;
    
    //(4.)return output
    return output_features;
}


fmap* Convolution::conv2d_optimized(fmap* input_features)
{
    //not done
    
    int start,end;
    start=clock();
    
    //declare all parameters
    int N=input_features->dim1;
    int C=input_features->dim2;
    int W=input_features->dim3;
    int H=input_features->dim4;
    int E = int((H + 2*Py-S)/Sy + 1);
    int F = int((W + 2*Px-R)/Sx + 1);
    
    //padding, H_ and W_ are the new dimensions
    int H_ = int(H + Py*2);
    int W_ = int(W + Px*2);
    fmap* input_new = (fmap*) malloc(sizeof(create_fmap(N, C , H_, W_)));
    *input_new = create_fmap(N, C, H_, W_);
    DATA (*temp_inputs_new)[C][H_][W_] = (DATA (*)[C][H_][W_])input_new->data;
    DATA (*temp_inputs)[C][H][W] = (DATA (*)[C][H][W])input_features->data;
    for(int n=0; n<N; n++){
        for(int c=0;c<C;c++){
            for(int y=0;y<H_;y++){
                for(int x=0;x<W_;x++){
                    if((x>=Px && x<F-Px) && (y>=Py && y<E-Py)){
                        temp_inputs_new[n][c][y][x] = 
                            temp_inputs[n][c][y-Py][x-Px];
                    }
                    else{
                        temp_inputs_new[n][c][y][x] = 0;
                    }
                }
            }
        }
    }
    free(input_features); //free memory allocated for older input features
    //std::cout<<"padding done"<<std::endl;
    
    //(1.)output feature map allocation
    fmap* output_features = (fmap*) malloc(sizeof(create_fmap(N, M, E, F)));
    *output_features = create_fmap(N, M, E, F);
    DATA (*temp_outputs)[M][E][F] = (DATA (*)[M][E][F])output_features->data;
    DATA (*temp_weights)[C][R][S] = (DATA (*)[C][R][S])weights;    

    //(2.)computation with for loop
    __m256 mm_inputs, mm_weights, product,sum;
    __m256i mask;
    float array[8];
    for(int n=0; n<N; n++){
        for(int m=0; m<M; m++){
            for(int f=0; f<F;f+=1){
                for(int e=0; e<E; e+=1){
                    temp_outputs[n][m][e][f] = 0;
                    sum=_mm256_setzero_ps();
                    for(int c=0; c<C; c++){
                        for(int r=0; r<R; r++){
                            for(int s=0; s<S/8; s++){   
                                int index = S-s*8-8;
                                mask=_mm256_set_epi32(index+8,index+7, index+6, index+5, 
                                                      index+4,index+3, index+2,index+1);
                                mm_inputs=_mm256_maskload_ps((float const*) &temp_inputs_new[n][c][Sy*e+r][Sx*f+s*8], mask);
                                mm_weights=_mm256_maskload_ps((float const*) &temp_weights[m][c][r][s*8], mask);
                                product=_mm256_mul_ps(mm_inputs, mm_weights);
                                sum=_mm256_add_ps(sum, product);
                            }
                        }
                    }
                    _mm256_storeu_ps((float*) &array,sum);
                    temp_outputs[n][m][e][f]=0;
                    for(int qa=0;qa<8;qa++){
                        temp_outputs[n][m][e][f]+=array[qa];
                    }                    
                }
            }
        }
    }
                                  
    //(3.)free input feature memory
    free(input_new);
    end =clock();
    exec_time = double(end-start) / double(CLOCKS_PER_SEC);
    //std::cout<<"conv done"<<std::endl;
    
    //(4.)return output
    return output_features;
}


fmap* Linear::linear(fmap* input_features)
{
    //does fc operation without any avx instructions
    
    //for time calculation
    int start,end;
    start=clock();
    
    //declare all parameters
    int N=input_features->dim1;

    //(1.)output feature map allocation
    fmap* output_features = (fmap*) malloc(sizeof(create_fmap(N, L, 1, 1)));
    *output_features      = create_fmap(N, L, 1, 1);
    DATA (*temp_outputs)[L] = (DATA (*)[L])output_features->data;
    DATA (*temp_weights)[L] = (DATA (*)[L])weights;
    DATA (*temp_inputs)[M] = (DATA (*)[M])input_features->data;
        
    //(2.)computation with for loop
    for(int n=0; n<N; n++){
        for(int l=0; l<L; l++){
            temp_outputs[n][l]=0;
            for(int m=0; m<M; m++){
                temp_outputs[n][l] += temp_weights[m][l]*temp_inputs[n][m];
            }
        }
    }

    //(3.) free input feature memory
    free(input_features);
    end =clock();
    exec_time = double(end-start) / double(CLOCKS_PER_SEC);
    //std::cout<<"fc done"<<std::endl;
    
    //(4.) return output
    return output_features;
}

fmap* Linear::linear_optimized(fmap* input_features)
{
  return NULL;
}

void relu(fmap* input_features)
{
    //we need input data only, modify and return no avx
    DATA (*temp) = (DATA (*))input_features->data;  
    int size = input_features->dim1*input_features->dim2*
        input_features->dim3*input_features->dim4;
    
    //modification for loop
    for(int i=0; i<size; i++){
        if(temp[i]<=0){
            temp[i] = 0;
        }
    }
    //std::cout<<"relu done"<<std::endl;
}

fmap* maxpool_2d(fmap* input_features, int Wx, int Wy, int Sx, int Sy)
{
    //does maxpooling without any avx instructions
    
    //for time calculation
    int start,end;
    start=clock();
    
    //declare all parameters
    int N=input_features->dim1;
    int C=input_features->dim2;
    int W=input_features->dim3;
    int H=input_features->dim4;
    int E = int((H-Wy)/Sy + 1);
    int F = int((W-Wx)/Sx + 1);
     
    //1.output feature map allocation
    //create_fmap creates an fmap with the given parameters
    fmap* output_features = (fmap*) malloc(sizeof(create_fmap(N, C, E, F)));
    *output_features = create_fmap(N, C, E, F);
    DATA (*temp_outputs)[C][E][F] = (DATA (*)[C][E][F])output_features->data;
    DATA (*temp_inputs)[C][H][W] = (DATA (*)[C][H][W])input_features->data;

    //2.computation with for loop using avx
    float array[8];
    __m256 v1,v2,v3,v4,v5,v6,v7,a,b;
    for(int n=0; n<N; n++){
        for(int c=0; c<C; c++){
            for(int e=0; e<E; e++){
                for(int f=0; f<F; f++){
                int max = 0;
                    for(int y=0; y<Wy; y++){
                        for(int x=0; x<Wx/16; x++){
                            //147 is control code for rotate left by upper 4 elements and lower 4 elements 
                            a = _mm256_loadu_ps((float const*) &temp_inputs[n][c][Wy*e+y][Wx*f+x*8]);
                            a = _mm256_loadu_ps((float const*) &temp_inputs[n][c][Wy*e+y][Wx*f+x*8+16]);
                            v1 = v3 = _mm256_max_ps(a,b);
                            v2 = _mm256_permute_ps(v1,(int)147);
                            v3 = _mm256_max_ps(v1,v2);
                            v4 = _mm256_permute_ps(v3,(int)147);
                            v5 = _mm256_max_ps(v3,v4);
                            v6 = _mm256_permute_ps(v5,(int)147);
                            v7 = _mm256_max_ps(v5,v6);
                            _mm256_storeu_ps((float*) &array,v7);
                            if(array[7]>max){
                                max=array[7];
                            }                          
                        }
                    }
                    temp_outputs[n][c][e][f] = max;
                }
            }
        }
    }
    
    //3. free input feature memory and calculate time
    free(input_features);
    //std::cout<<"maxpool done"<<std::endl;
    
    //4. return output
    return output_features;
}

AlexNet::AlexNet()
{
  conv_layers = (Convolution**) malloc(5 * sizeof(Convolution*));

  Convolution *conv;
  conv = new Convolution(96, 3, 11, 11, 4, 4, 0, 0);
  conv_layers[0] = conv;
  conv = new Convolution(256, 96, 5, 5, 1, 1, 2, 2);
  conv_layers[1] = conv;
  conv = new Convolution(384, 256, 3, 3, 1, 1, 1, 1);
  conv_layers[2] = conv;
  conv = new Convolution(384, 384, 3, 3, 1, 1, 1, 1);
  conv_layers[3] = conv;
  conv = new Convolution(256, 384, 3, 3, 1, 1, 1, 1);
  conv_layers[4] = conv;

  linear_layers = (Linear**) malloc(3 * sizeof(Linear*));

  Linear *linear;
  linear = new Linear(4096, 9216);
  linear_layers[0] = linear;
  linear = new Linear(4096, 4096);
  linear_layers[1] = linear;
  linear = new Linear(1000, 4096);
  linear_layers[2] = linear;
}

fmap* AlexNet::forward_pass(fmap* input_features)
{
  int start, end;
  start = clock();

  fmap* temp = input_features;
  
  temp = conv_layers[0]->conv2d_IS(temp);  //conv_2d conv2d_optimized conv2d_WS
  relu(temp);
  temp = maxpool_2d(temp, 3, 3, 2, 2);
  temp = conv_layers[1]->conv2d_IS(temp);
  relu(temp);
  temp = maxpool_2d(temp, 3, 3, 2, 2);
  temp = conv_layers[2]->conv2d_IS(temp);
  relu(temp);
  temp = conv_layers[3]->conv2d_IS(temp);
  relu(temp);
  temp = conv_layers[4]->conv2d_IS(temp);
  relu(temp);
  temp = maxpool_2d(temp, 3, 3, 2, 2);

  int lin_dim = temp->dim2 * temp->dim3 * temp->dim4;
  temp->dim2 = lin_dim;
  temp->dim3 = temp->dim4 = 1;

  temp = linear_layers[0]->linear(temp);
  relu(temp);
  temp = linear_layers[1]->linear(temp);
  relu(temp);
  temp = linear_layers[2]->linear(temp);
  relu(temp);

  end = clock();

  exec_time = double(end-start) / double(CLOCKS_PER_SEC);
  return temp;
}
