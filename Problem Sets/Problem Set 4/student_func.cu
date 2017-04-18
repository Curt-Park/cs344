//Udacity HW 4
//Radix Sorting

#include "reference_calc.cpp"
#include "utils.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <device_launch_parameters.h>
#include <device_functions.h>
#include <iostream>
#include <stdio.h>

/* Red Eye Removal
   ===============
   
   For this assignment we are implementing red eye removal.  This is
   accomplished by first creating a score for every pixel that tells us how
   likely it is to be a red eye pixel.  We have already done this for you - you
   are receiving the scores and need to sort them in ascending order so that we
   know which pixels to alter to remove the red eye.

   Note: ascending order == smallest to largest

   Each score is associated with a position, when you sort the scores, you must
   also move the positions accordingly.

   Implementing Parallel Radix Sort with CUDA
   ==========================================

   The basic idea is to construct a histogram on each pass of how many of each
   "digit" there are.   Then we scan this histogram so that we know where to put
   the output of each digit.  For example, the first 1 must come after all the
   0s so we have to know how many 0s there are to be able to start moving 1s
   into the correct position.

   1) Histogram of the number of occurrences of each digit
   2) Exclusive Prefix Sum of Histogram
   3) Determine relative offset of each digit
        For example [0 0 1 1 0 0 1]
                ->  [0 1 0 1 2 3 2]
   4) Combine the results of steps 2 & 3 to determine the final
      output location for each element and move it there

   LSB Radix sort is an out-of-place sort and you will need to ping-pong values
   between the input and output buffers we have provided.  Make sure the final
   sorted results end up in the output buffer!  Hint: You may need to do a copy
   at the end.

 */
__global__ void generate_histogram(const unsigned int* const d_inputVals, 
                                   unsigned int* const d_binHistogram,
                                   const unsigned int mask,
                                   const unsigned int shift,
                                   const size_t numElems)
{
  const int idx = blockIdx.x*blockDim.x+threadIdx.x;
  
  if(idx >= numElems)
    return;
    
  const unsigned int bin = (d_inputVals[idx] & mask) >> shift;

  atomicAdd(&d_binHistogram[bin], 1);
}

__global__ void exclusive_prefix_sum(const unsigned int* const d_binHistogram,
                                     unsigned int* const d_binScan, 
                                     const size_t numBins)
{
  extern __shared__ unsigned int temp_binScan[];
  
  unsigned int idx = threadIdx.x;
  
  int offset = 1;
  temp_binScan[2*idx] = d_binHistogram[2*idx];
  temp_binScan[2*idx+1] = d_binHistogram[2*idx+1];
  
  for(int d = numBins>>1; d > 0; d >>= 1)
  {
    __syncthreads();
    if(idx < d)
    {
      int ai = offset*(2*idx+1)-1;
      int bi = offset*(2*idx+2)-1;
      temp_binScan[bi] += temp_binScan[ai]; 
    }
    offset *= 2;
  }
  
  if(idx == 0) { temp_binScan[numBins - 1] = 0;} // clear the last element

  for (int d = 1; d < numBins; d *= 2) // traverse down tree & build scan
  {
    offset >>= 1;
    __syncthreads();
    
    if(idx < d)
    {
      int ai = offset*(2*idx+1)-1;
      int bi = offset*(2*idx+2)-1;
    
      float t = temp_binScan[ai]; 
      temp_binScan[ai] = temp_binScan[bi];
      temp_binScan[bi] += t;
    }
  }
  __syncthreads();

  d_binScan[2*idx] = temp_binScan[2*idx]; // write results to device memory
  d_binScan[2*idx+1] = temp_binScan[2*idx+1];
}

__global__ void generate_flag(const unsigned int* const d_inputVals, 
                              unsigned int* const d_segmentedScan,
                              const unsigned int mask,
                              const unsigned int shift,
                              const size_t numElems)
{
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx >= numElems)
    return;
    
  const unsigned int bin = (d_inputVals[idx] & mask) >> shift;
  d_segmentedScan[bin*numElems+idx] = 1;
}

__global__ void gather_into_correct_location(const unsigned int* const d_inputVals,
                                             const unsigned int* const d_inputPos,
                                             unsigned int* const d_outputVals,
                                             unsigned int* const d_outputPos,
                                             const unsigned int* const d_binScan,
                                             const unsigned int* const d_segmentedScan,
                                             const unsigned int mask,
                                             const unsigned int shift,
                                             const size_t numElems)
{
  const int idx = blockIdx.x*blockDim.x+threadIdx.x;
  
  if(idx >= numElems)
    return;
 
  const unsigned int bin = (d_inputVals[idx] & mask) >> shift;
  const unsigned int pos = d_binScan[bin] + d_segmentedScan[bin*numElems+idx] -1;
  
  d_outputVals[pos] = d_inputVals[idx];
  d_outputPos[pos]  = d_inputPos[idx];
}

void swap(unsigned int** p1, unsigned int** p2)
{
  unsigned int* temp=*p1;
  *p1 = *p2;
  *p2 = temp;
}

void test_print(const unsigned int* const d_vals, size_t num)
{
    unsigned int* temp = (unsigned int*)malloc(sizeof(unsigned int)*num);
    checkCudaErrors(cudaMemcpy(temp, d_vals, sizeof(unsigned int)*num, cudaMemcpyDeviceToHost));
    for(unsigned int i=0; i<num; ++i)
        printf("%d ", temp[i]);
    puts("");
    free(temp);
}
/*
void your_sort(unsigned int* const d_inputVals,
               unsigned int* const d_inputPos,
               unsigned int* const d_outputVals,
               unsigned int* const d_outputPos,
               const size_t numElems)
{
  const dim3 blockSize(256);
  const dim3 gridSize((numElems+blockSize.x-1)/blockSize.x);
  
  const int numBits = 1;
  const int numBins = 1 << numBits;
  
  unsigned int *d_binHistogram;
  unsigned int *d_binScan;
  
  unsigned int *vals_src = d_inputVals;
  unsigned int *pos_src  = d_inputPos;

  unsigned int *vals_dst = d_outputVals;
  unsigned int *pos_dst  = d_outputPos;
  
  checkCudaErrors(cudaMalloc(&d_binHistogram, sizeof(unsigned int)*numBins));
  checkCudaErrors(cudaMalloc(&d_binScan, sizeof(unsigned int)*numBins));
  thrust::device_vector<unsigned int> d_segmentedScan(numElems*2);
  
   //a simple radix sort - only guaranteed to work for numBits that are multiples of 2
  for (unsigned int i = 0; i < 8 * sizeof(unsigned int); i += numBits) {
    unsigned int mask = (numBins - 1) << i;

    checkCudaErrors(cudaMemset(d_binHistogram, 0, sizeof(unsigned int)*numBins)); //zero out the bins
    checkCudaErrors(cudaMemset(d_binScan, 0, sizeof(unsigned int)*numBins)); //zero out the bins
    checkCudaErrors(cudaMemset(thrust::raw_pointer_cast(&d_segmentedScan[0]), 0, sizeof(unsigned int)*numElems*2)); //zero out the bins

    //perform histogram of data & mask into bins
    generate_histogram<<<gridSize, blockSize>>>(vals_src, d_binHistogram, mask, i, numElems);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

    //perform exclusive prefix sum (scan) on binHistogram to get starting
    //location for each bin
    exclusive_prefix_sum<<<1, numBins/2, sizeof(unsigned int)*numBins>>>(d_binHistogram, d_binScan, numBins);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
    
    //perform generating segmented scan
    generate_flag<<<gridSize, blockSize>>>(d_inputVals, 
                                           thrust::raw_pointer_cast(&d_segmentedScan[0]),
                                           mask, i, numElems);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
    thrust::inclusive_scan(d_segmentedScan.begin(), d_segmentedScan.begin()+numElems, d_segmentedScan.begin());
    thrust::inclusive_scan(d_segmentedScan.begin()+numElems, d_segmentedScan.end(), d_segmentedScan.begin()+numElems);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

    //Gather everything into the correct location
    //need to move vals and positions
    gather_into_correct_location<<<gridSize, blockSize>>>(vals_src, pos_src, 
                                                              vals_dst, pos_dst, 
                                                              d_binScan,
                                                              thrust::raw_pointer_cast(&d_segmentedScan[0]),
                                                              mask, i, numElems);
    cudaDeviceSynchronize(); //checkCudaErrors(cudaGetLastError());
    
    swap(&vals_dst, &vals_src);
    swap(&pos_dst, &pos_src);
  }
  
  checkCudaErrors(cudaMemcpy(d_outputVals, d_inputVals, sizeof(unsigned int)*numElems, cudaMemcpyDeviceToDevice));
  checkCudaErrors(cudaMemcpy(d_outputPos, d_inputPos, sizeof(unsigned int)*numElems, cudaMemcpyDeviceToDevice));

  checkCudaErrors(cudaFree(d_binHistogram));
  checkCudaErrors(cudaFree(d_binScan));
}
*/

void your_sort(unsigned int* const d_inputVals,
               unsigned int* const d_inputPos,
               unsigned int* const d_outputVals,
               unsigned int* const d_outputPos,
               const size_t numElems)
{

  // Thrust vectors wrapping raw GPU data
  thrust::device_ptr<unsigned int> d_inputVals_p(d_inputVals);
  thrust::device_ptr<unsigned int> d_inputPos_p(d_inputPos);
  thrust::host_vector<unsigned int> h_inputVals_vec(d_inputVals_p, d_inputVals_p + numElems);
  thrust::host_vector<unsigned int> h_inputPos_vec(d_inputPos_p, d_inputPos_p + numElems);

  thrust::sort_by_key(h_inputVals_vec.begin(), h_inputVals_vec.end(), h_inputPos_vec.begin());
  checkCudaErrors(cudaMemcpy(d_outputVals, thrust::raw_pointer_cast(&h_inputVals_vec[0]),
                             numElems * sizeof(unsigned int), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_outputPos, thrust::raw_pointer_cast(&h_inputPos_vec[0]),
                             numElems * sizeof(unsigned int), cudaMemcpyHostToDevice));
}