/*
* a kernel that add the elements of two vectors pairwise
*/
__kernel void gaussian_blur(
	__global const unsigned char *A,
	__global const float *B,
	__global unsigned char *C,
	int width,
	int height)
{
	size_t id = get_global_id(0);
	size_t localId = get_local_id(0);
	size_t localSize = get_local_size(0);

	//printf("Global: %d Local: %d LocalSize: %d\n", id, localId, localSize);
	 
	int row = id / width;
	int col = id % width;

	// omit alpha channel
	if (id % 4 == 3)
	{
		C[id] = A[id];
		return;
	}

	// omit border
	if (col < 4 || col >= width - 4 || row == 0 || row >= height - 1)
	{
		C[id] = A[id];
		return;
	}

	 //printf("Apply (%d | %d)\n", row, col);

	 // calculate value for pixel by matrix mult
	 float sum = 0.0f;
	 sum += A[id - width - 1] * B[8];
	 sum += A[id - width] * B[7];
	 sum += A[id - width + 1] * B[6];
	 sum += A[id - 1] * B[5];
	 sum += A[id] * B[4];
	 sum += A[id + 1] * B[3];
	 sum += A[id + width - 1] * B[2];
	 sum += A[id + width] * B[1];
	 sum += A[id + width + 1] * B[0];

	 //printf("Calculated %f\n", sum);

	 C[id] = sum;
}