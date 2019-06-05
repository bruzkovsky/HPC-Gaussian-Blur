/*
* a kernel that add the elements of two vectors pairwise
*/
__kernel void gaussian_blur(
	__global const unsigned char *A,
	__global const float *B,
	__global unsigned char *C,
	int width,
	int height,
	int filterWidth,
	int filterHeight)
{
	size_t id = get_global_id(0);
	size_t localId = get_local_id(0);
	size_t localSize = get_local_size(0);
	size_t groupId = get_group_id(0);
	size_t numGroups = get_num_groups(0);

	//printf("Global: %d Local: %d LocalSize: %d GroupId: %d NumGroups: %d\n", id, localId, localSize, groupId, numGroups);
	 
	int row = id / width;
	int col = id % width;

	// omit alpha channel
	if (id % 4 == 3)
	{
		C[id] = A[id];
		return;
	}

	// omit border
	if (col < 4 || col >= width - 4 || row == 0 || row >= height / 4 - 1)
	{
		C[id] = A[id];
		return;
	}

	 //printf("Apply (%d | %d)\n", row, col);

	 // calculate value for pixel by matrix mult
	 float sum = 0.0f;

	 if (filterWidth == filterHeight)
	 {
		// one-pass
		 for (unsigned int i = 0; i < filterWidth * filterHeight; i++)
		 {
			 int posX = filterWidth / 2 - (i % filterWidth);
			 int posY = filterHeight / 2 - i / filterHeight;
			 //printf("%d: %d|%d\n", id, posX, posY);
			 sum += A[id + width * posY + 4 * posX] * B[i];
		 }
	 }
	 else if (filterHeight == 1)
	 {
		//two-pass horizontal
		 for (unsigned int i = 0; i < filterWidth; i++)
		 {
			 int pos = filterWidth / 2 - i;
			 //printf("%d: %d\n", id, pos);
			 sum += A[id + 4 * pos] * B[i];
		 }
	 }
	 else if (filterWidth == 1)
	 {
		//two-pass vertical
		 for (unsigned int i = 0; i < filterHeight; i++)
		 {
			 int pos = filterHeight / 2 - i;
			 //printf("%d: %d\n", id, pos);
			 sum += A[id + width * pos] * B[i];
		 }
	 }
	 else return;

	 //printf("Calculated %f\n", sum);

	 C[id] = sum;
}
