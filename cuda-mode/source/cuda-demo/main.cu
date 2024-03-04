#include <stdio.h>
#include <cuda_fp16.h>
#define DSIZE 8
#define SCF 0.5f
#define nTPB 256
__global__ void half_scale_kernel(float *din, float *dout, int dsize){

  int idx = threadIdx.x+blockDim.x*blockIdx.x;
  if (idx < dsize){
    half scf = __float2half(SCF);
    half kin = __float2half(din[idx]);
    half kout;
    

#if __CUDA_ARCH__ >= 530
    printf("__CUDA_ARCH__ >= 530\n");
    kout = __hmul(kin, scf);
#else
    kout = __float2half(__half2float(kin)*__half2float(scf));
#endif
    dout[idx] = __half2float(kout);
    }
}

int main(){

  float *hin, *hout, *din, *dout;
  hin  = (float *)malloc(DSIZE*sizeof(float));
  hout = (float *)malloc(DSIZE*sizeof(float));
  for (int i = 0; i < DSIZE; i++) hin[i] = i;
  cudaMalloc(&din,  DSIZE*sizeof(float));
  cudaMalloc(&dout, DSIZE*sizeof(float));
  cudaMemcpy(din, hin, DSIZE*sizeof(float), cudaMemcpyHostToDevice);
  half_scale_kernel<<<(DSIZE+nTPB-1)/nTPB,nTPB>>>(din, dout, DSIZE);
  cudaMemcpy(hout, dout, DSIZE*sizeof(float), cudaMemcpyDeviceToHost);
  for (int i = 0; i < DSIZE; i++) printf("%f\n", hout[i]);
  return 0;
}