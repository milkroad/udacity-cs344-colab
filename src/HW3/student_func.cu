/* Udacity Homework 3
   HDR Tone-mapping

  Background HDR
  ==============

  A High Dynamic Range (HDR) image contains a wider variation of intensity
  and color than is allowed by the RGB format with 1 byte per channel that we
  have used in the previous assignment.  

  To store this extra information we use single precision floating point for
  each channel.  This allows for an extremely wide range of intensity values.

  In the image for this assignment, the inside of church with light coming in
  through stained glass windows, the raw input floating point values for the
  channels range from 0 to 275.  But the mean is .41 and 98% of the values are
  less than 3!  This means that certain areas (the windows) are extremely bright
  compared to everywhere else.  If we linearly map this [0-275] range into the
  [0-255] range that we have been using then most values will be mapped to zero!
  The only thing we will be able to see are the very brightest areas - the
  windows - everything else will appear pitch black.

  The problem is that although we have cameras capable of recording the wide
  range of intensity that exists in the real world our monitors are not capable
  of displaying them.  Our eyes are also quite capable of observing a much wider
  range of intensities than our image formats / monitors are capable of
  displaying.

  Tone-mapping is a process that transforms the intensities in the image so that
  the brightest values aren't nearly so far away from the mean.  That way when
  we transform the values into [0-255] we can actually see the entire image.
  There are many ways to perform this process and it is as much an art as a
  science - there is no single "right" answer.  In this homework we will
  implement one possible technique.

  Background Chrominance-Luminance
  ================================

  The RGB space that we have been using to represent images can be thought of as
  one possible set of axes spanning a three dimensional space of color.  We
  sometimes choose other axes to represent this space because they make certain
  operations more convenient.

  Another possible way of representing a color image is to separate the color
  information (chromaticity) from the brightness information.  There are
  multiple different methods for doing this - a common one during the analog
  television days was known as Chrominance-Luminance or YUV.

  We choose to represent the image in this way so that we can remap only the
  intensity channel and then recombine the new intensity values with the color
  information to form the final image.

  Old TV signals used to be transmitted in this way so that black & white
  televisions could display the luminance channel while color televisions would
  display all three of the channels.
  

  Tone-mapping
  ============

  In this assignment we are going to transform the luminance channel (actually
  the log of the luminance, but this is unimportant for the parts of the
  algorithm that you will be implementing) by compressing its range to [0, 1].
  To do this we need the cumulative distribution of the luminance values.

  Example
  -------

  input : [2 4 3 3 1 7 4 5 7 0 9 4 3 2]
  min / max / range: 0 / 9 / 9

  histo with 3 bins: [4 7 3]

  cdf : [4 11 14]


  Your task is to calculate this cumulative distribution by following these
  steps.

*/

#include "utils.h"

__global__ 
void reduction_kernel(float* d_in, float* d_out, const int type, const int boundary)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (unsigned int s=blockDim.x/2; s>0; s>>=1) {
        if (threadIdx.x < s && idx < (boundary - s)) {
            if (type==0) { // min
               d_in[idx] = fminf(d_in[idx],d_in[idx+s]);
            }else{ // max
               d_in[idx] = fmaxf(d_in[idx],d_in[idx+s]);
               //if (threadIdx.x==(s-1) && blockIdx.x==(gridDim.x-1)) {
               //    printf("\tdebug: max of [%d], [%d] is %f\n", idx, idx+s, fmaxf(d_in[idx],d_in[idx+s]) );
               //}
            }
        }
        __syncthreads();
    }
    
    if (threadIdx.x==0) {
       d_out[blockIdx.x] = d_in[idx];
    }
}

float reduce(const float* const d_logLuminance, 
            int size, int type)
{
   const int ARRAY_BYTES = size * sizeof(float);
   float * d_in, * d_intermediate, * d_out;
   
   checkCudaErrors(cudaMalloc(&d_in, ARRAY_BYTES));
   checkCudaErrors(cudaMalloc(&d_intermediate, ARRAY_BYTES));
   checkCudaErrors(cudaMalloc(&d_out, ARRAY_BYTES));
   cudaMemcpy(d_in, d_logLuminance, ARRAY_BYTES, cudaMemcpyDeviceToDevice); 
   
   const int maxThreadsPerBlock = 1024;
   int threads = maxThreadsPerBlock;
   int blocks = ceil(float(size)/float(threads));

   printf("Reduction Kernel launch: <<<%d,%d>>>\n", blocks, threads);
   reduction_kernel<<<blocks, threads, threads * sizeof(float)>>>
       (d_in, d_intermediate, type, size);
   //float *h_test = (float*) malloc(size * sizeof(float));
   //cudaMemcpy(h_test, d_intermediate, size * sizeof(float), cudaMemcpyDeviceToHost);
   //printf("debug: intermediate max: %f\n", h_test[blocks-1]);

   // now we're down to one block left, so reduce it
   threads = 1024; // always launch 1024 threads, the kernel will take care of the boudary check
   blocks = 1; // TODO
   printf("Reduction Kernel launch: <<<%d,%d>>>\n", blocks, threads);
   reduction_kernel<<<blocks, threads, threads * sizeof(float)>>>
       (d_intermediate, d_out, type, size);

   float h_out;
   cudaMemcpy(&h_out, d_out, sizeof(float), cudaMemcpyDeviceToHost);

   cudaFree(d_in);
   cudaFree(d_out);
   
   return h_out;
}

__global__ 
void atomic_histo_kernel(unsigned int *d_bins, const float *d_in, const int BIN_COUNT, 
                         const float lumRange, const float lumMin, const int boudary)
{
   int idx = blockIdx.x * blockDim.x + threadIdx.x;
   if (idx < boudary) {
      int myBin = (d_in[idx] - lumMin) / lumRange * BIN_COUNT;
      atomicAdd(&(d_bins[myBin]), 1);
   }

}

__global__ 
void HnS_scan_kernel(unsigned int *d_in, unsigned int *d_out, const int size)
{
   int idx = blockIdx.x * blockDim.x + threadIdx.x;
   for (int s = 1; s <= size / 2; s<<=1) {
      if (idx < size - s) {
         int sum = d_in[idx] + d_in[idx+s];
         //if (idx==size-s-1) {
         //   printf("[%d]+[%d] = %d + %d = \n", idx, idx+s, d_in[idx], d_in[idx+s]);
         //}
         __syncthreads();
         d_in[idx+s] = sum;
         //if (idx==size-s-1) {
         //   printf("%d\n", sum);
         //}
         //printf("[%d]+[%d] = %d\n", idx, idx+s, d_in[idx+s]);
        //if(idx==0) {
        //   printf("idx=7: %d\n", d_in[7]);
        //}
        __syncthreads();
      }
   }
   
   if(idx < size/2) {
      d_out[idx+1] = d_in[idx];
      d_out[idx+1+size/2] = d_in[idx+size/2];
   //   printf("[%d] %d=%d\n", idx, d_out[idx], d_in[idx]);
   }
   if(idx==0) {
      d_out[idx] = 0;
   }
}

void your_histogram_and_prefixsum(const float* const d_logLuminance,
                                  unsigned int* const d_cdf,
                                  float &min_logLum,
                                  float &max_logLum,
                                  const size_t numRows,
                                  const size_t numCols,
                                  const size_t numBins)
{
   // min/max reduction:
   size_t numPixels = numRows * numCols;
   float lumRange;
   
   min_logLum = reduce(d_logLuminance, numPixels, 0);
   max_logLum = reduce(d_logLuminance, numPixels, 1);
   lumRange = max_logLum - min_logLum;
   printf("min: %f\tmax: %f\n", min_logLum, max_logLum);

   // hostogram:
   unsigned int *d_bins;
   checkCudaErrors(cudaMalloc(&d_bins, numBins * sizeof(int)));
   int blocks = ceil(float(numPixels)/float(1024));
   printf("Histo Kernel launch: <<<%d,1024>>>\n", blocks);
   atomic_histo_kernel<<<blocks, 1024>>>(d_bins, d_logLuminance, numBins, lumRange, min_logLum, numPixels);
   int *h_test = (int*) malloc(numBins * sizeof(int));
   cudaMemcpy(h_test, d_bins, numBins * sizeof(int), cudaMemcpyDeviceToHost);
   printf("GPU: ");
   for (size_t i = 0; i < numBins; ++i) {
      printf("%d ", h_test[i]);
      if ((i%30)==1) {
         printf("\n");
      }
   }
   printf("\n");

   // exclusive scan(prefix sum)
   printf("Scan Kernel launch: <<<%d,%d>>>\n", 1, numBins);
   HnS_scan_kernel<<<1, numBins>>>(d_bins, d_cdf, numBins);
   cudaMemcpy(h_test, d_cdf, numBins * sizeof(int), cudaMemcpyDeviceToHost);
   printf("GPU: ");
   for (size_t i = 0; i < numBins; ++i) {
      printf("%d ", h_test[i]);
      if ((i%30)==1) {
         printf("\n");
      }
   }
   printf("\n");
   
   /*Here are the steps you need to implement
     1) find the minimum and maximum value in the input logLuminance channel
        store in min_logLum and max_logLum
     2) subtract them to find the range
     3) generate a histogram of all the values in the logLuminance channel using
        the formula: bin = (lum[i] - lumMin) / lumRange * numBins
     4) Perform an exclusive scan (prefix sum) on the histogram to get
        the cumulative distribution of luminance values (this should go in the
        incoming d_cdf pointer which already has been allocated for you)       */


}
