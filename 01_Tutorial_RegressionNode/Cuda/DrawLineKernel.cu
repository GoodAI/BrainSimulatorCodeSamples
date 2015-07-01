#include <cuda.h> 
#include <device_launch_parameters.h> 

// draw line as y = k*x+q
extern "C"
{
	__constant__ int D_SIZE;
	__constant__ float D_K;
	__constant__ float D_Q;
	//__constant__ float D_SCALE;
	__constant__ float D_XSCALE;
	__constant__ float D_YSCALE;
	__constant__ float D_XMIN;
	__constant__ float D_YMIN;

	__global__ void DrawLineKernel(unsigned int *pixels, int count)
	{
		int threadId = blockDim.x*blockIdx.y*gridDim.x
			+ blockDim.x*blockIdx.x
			+ threadIdx.x;

		if (threadId < count)
		{
			int xvalue = threadId;
			float altx = (xvalue / D_XSCALE) + D_XMIN;

			float yvalue = D_K * altx + D_Q;
			int ynew = (int)((yvalue - D_YMIN) * D_YSCALE);

			int pixy = D_SIZE - ynew;
			if (pixy >= D_SIZE || pixy <= 0){
				return;
			}

			int idx = pixy * D_SIZE + xvalue;

			pixels[idx] = 0xFFFF0000;
		}
	}
}