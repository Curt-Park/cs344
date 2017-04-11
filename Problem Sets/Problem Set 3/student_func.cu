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


#include "reference_calc.cpp"
#include "utils.h"
#include <stdio.h>

// Hillis and Steele 
__global__ void reduce_min_max_kernel(const float* const d_logLuminance,
                                      float* const d_out,
                                      const size_t numRows,
                                      const size_t numCols,
                                      const bool is_min)
{
  extern __shared__ float temp[];
  
  const int2 idx_2d = make_int2(blockIdx.x*blockDim.x+threadIdx.x, blockIdx.y*blockDim.y+threadIdx.y);
  const int idx = idx_2d.y*numCols + idx_2d.x;
  const int temp_idx = blockDim.x*threadIdx.y + threadIdx.x;
  const int temp_size = blockDim.x*blockDim.y;
  int pout=0, pin=1; // for double bufferring
  
  if(idx_2d.x >= numCols || idx_2d.y >= numRows)
    temp[pout*temp_size + temp_idx] = (is_min-1)*255.0f;
  else
    temp[pout*temp_size + temp_idx] = d_logLuminance[idx];
  __syncthreads();
  
  for(int offset=1; offset<temp_size; offset*=2)
  {
    pout = 1 - pout; // 1 <-> 0
    pin = 1 - pout; // 0 <-> 1
    
    if(temp_idx-offset >= 0)
      temp[pout*temp_size + temp_idx] = is_min==true? min(temp[pin*temp_size + temp_idx],
                                                          temp[pin*temp_size + temp_idx-offset]):
                                                      max(temp[pin*temp_size + temp_idx], 
                                                          temp[pin*temp_size + temp_idx-offset]);
    else
      temp[pout*temp_size + temp_idx] = temp[pin*temp_size + temp_idx];
    __syncthreads();
  }
  
  if(temp_idx == temp_size-1)
  {
    d_out[gridDim.x*blockIdx.y + blockIdx.x] = temp[pout*temp_size + temp_idx];
  }
}

__global__ void generate_histogram_kernel(const float* const d_logLuminance,
                                          unsigned int* const d_histo,
                                          const size_t numRows, 
                                          const size_t numCols, 
                                          const size_t numBins,
                                          const float min_logLum,
                                          const float range_logLum)
{
  const int2 idx_2d = make_int2(blockIdx.x*blockDim.x+threadIdx.x, blockIdx.y*blockDim.y+threadIdx.y);
  const int idx = idx_2d.y*numCols + idx_2d.x;
  
  if(idx_2d.x >= numCols || idx_2d.y >= numRows)
    return;
    
  unsigned int bin = min((unsigned int)(numBins - 1),
                         (unsigned int)((d_logLuminance[idx] - min_logLum) / range_logLum * numBins));

  atomicAdd(&d_histo[bin], 1);
}

__global__ void generate_cdf_kernel(const unsigned int* const d_histo,
                                    unsigned int* const d_cdf, 
                                    const size_t numBins)
{
  extern __shared__ unsigned int temp_cdf[];
  
  unsigned int idx = threadIdx.x;
  
  int offset = 1;
  temp_cdf[2*idx] = d_histo[2*idx];
  temp_cdf[2*idx+1] = d_histo[2*idx+1];
  
  for(int d = numBins>>1; d > 0; d >>= 1)
  {
    __syncthreads();
    if(idx < d)
    {
      int ai = offset*(2*idx+1)-1;
      int bi = offset*(2*idx+2)-1;
      temp_cdf[bi] += temp_cdf[ai]; 
    }
    offset *= 2;
  }
  
  if(idx == 0) { temp_cdf[numBins - 1] = 0;} // clear the last element

  for (int d = 1; d < numBins; d *= 2) // traverse down tree & build scan
  {
    offset >>= 1;
    __syncthreads();
    
    if(idx < d)
    {
      int ai = offset*(2*idx+1)-1;
      int bi = offset*(2*idx+2)-1;
    
      float t = temp_cdf[ai]; 
      temp_cdf[ai] = temp_cdf[bi];
      temp_cdf[bi] += t;
    }
  }
  __syncthreads();

  d_cdf[2*idx] = temp_cdf[2*idx]; // write results to device memory
  d_cdf[2*idx+1] = temp_cdf[2*idx+1];
}

float reduce_min_max(const float* const d_logLuminance,
                     const size_t numRows,
                     const size_t numCols,
                     const bool is_min)
{
  float ret=0.f;
  float* d_out;
  const dim3 blockSize(16,16);
  const dim3 gridSize((numCols+blockSize.x-1)/blockSize.x, (numRows+blockSize.y-1)/blockSize.y);

  checkCudaErrors(cudaMalloc(&d_out, sizeof(float)*gridSize.x*gridSize.y));
  checkCudaErrors(cudaMemset(d_out, 0, sizeof(float)*gridSize.x*gridSize.y));
  
  // reduce for each block
  reduce_min_max_kernel<<<gridSize, blockSize, 2*sizeof(float)*blockSize.x*blockSize.y>>>(d_logLuminance, 
                                                                                          d_out,
                                                                                          numRows, numCols,
                                                                                          is_min);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

  // reduce for all min values of each block
  reduce_min_max_kernel<<<1, gridSize, 2*sizeof(float)*gridSize.x*gridSize.y>>>(d_out,
                                                                                d_out,
                                                                                gridSize.y, gridSize.x, 
                                                                                is_min);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
  
  checkCudaErrors(cudaMemcpy(&ret, &d_out[0], sizeof(float), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaFree(d_out));
  
  return ret;
}

void generate_histogram(const float* const d_logLuminance,
                        unsigned int* const d_histo,
                        const size_t numRows, 
                        const size_t numCols, 
                        const size_t numBins,
                        const float min_logLum,
                        const float range_logLum)
{
  const dim3 blockSize(16,16);
  const dim3 gridSize((numCols+blockSize.x-1)/blockSize.x, (numRows+2*blockSize.y-1)/blockSize.y);
  
  generate_histogram_kernel<<<gridSize, blockSize>>>(d_logLuminance,
                                                     d_histo,
                                                     numRows, 
                                                     numCols,
                                                     numBins,
                                                     min_logLum,
                                                     range_logLum);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
}

void generate_cdf(const unsigned int* const d_histo,
                  unsigned int* const d_cdf, 
                  const size_t numBins)
{
  generate_cdf_kernel<<<1, numBins/2, sizeof(unsigned int)*numBins>>>(d_histo,
                                                                      d_cdf, 
                                                                      numBins);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
}

/*
// For testing
void print_min_max(const float* const d_logLuminance,
                   const size_t numRows,
                   const size_t numCols)
{
  // for checking
  float* h_logLuminance = (float*)malloc(numRows*numCols*sizeof(float));
  checkCudaErrors(cudaMemcpy(h_logLuminance,d_logLuminance,numRows*numCols*sizeof(float),
                             cudaMemcpyDeviceToHost));
                             
  // This particular image had 240 cols and 294 rows
  printf("\n\nREFERENCE CALCULATION (Min/Max): \n");
  float logLumMin = h_logLuminance[0];
  float logLumMax = h_logLuminance[0];
  for (size_t i = 1; i < numCols * numRows; ++i) {
    logLumMin = min(h_logLuminance[i], logLumMin);
    logLumMax = max(h_logLuminance[i], logLumMax);
  }
  printf("  Min logLum: %f\n  Max logLum: %f\n",logLumMin,logLumMax);
  free(h_logLuminance);
}
*/

void your_histogram_and_prefixsum(const float* const d_logLuminance,
                                  unsigned int* const d_cdf,
                                  float &min_logLum,
                                  float &max_logLum,
                                  const size_t numRows,
                                  const size_t numCols,
                                  const size_t numBins)
{
  // Step1: Min/Max
  max_logLum = reduce_min_max(d_logLuminance, numRows, numCols, false); //max
  min_logLum = reduce_min_max(d_logLuminance, numRows, numCols, true); //min
  printf("max_logLum: %f\n", max_logLum);
  printf("min_logLum: %f\n", min_logLum);
  
  // Step2: Range
  float range_logLum = max_logLum-min_logLum;
  printf("range_logLum: %f\n", range_logLum);
  
  // Step3: Histogram
  unsigned int* d_histo;
  checkCudaErrors(cudaMalloc(&d_histo, sizeof(unsigned int)*numBins));
  checkCudaErrors(cudaMemset(d_histo, 0, sizeof(unsigned int)*numBins));
  generate_histogram(d_logLuminance, d_histo, numRows, numCols, numBins, min_logLum, range_logLum);
  
  // Step4: CDF
  generate_cdf(d_histo, d_cdf, numBins);
  checkCudaErrors(cudaFree(d_histo));
  
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