#include <cuda.h> 
#include <device_launch_parameters.h> 

#define PIXEL_COLOR 0xFF585858;

extern "C"
{
	__constant__ int D_SIZE;
	__constant__ float D_ALPHA;
	__constant__ float D_BETA;
	//__constant__ float D_SCALE;
	__constant__ float D_XSCALE;
	__constant__ float D_YSCALE;
	__constant__ float D_XMIN;
	__constant__ float D_YMIN;

	__global__ void RegressionObserverKernel(float *xdata, float *ydata, unsigned int *pixels, int count)
	{
		int threadId = blockDim.x*blockIdx.y*gridDim.x
			+ blockDim.x*blockIdx.x
			+ threadIdx.x;

		if (threadId < count)
		{
			float xvalue = xdata[threadId];
			int xnew = (int)((xvalue - D_XMIN) * D_XSCALE);

			float yvalue = ydata[threadId];
			int ynew = (int)((yvalue - D_YMIN) * D_YSCALE);

			int pixy = D_SIZE - ynew;

			int idx = pixy * D_SIZE + xnew;

			pixels[idx] = PIXEL_COLOR;
		}
	}
}