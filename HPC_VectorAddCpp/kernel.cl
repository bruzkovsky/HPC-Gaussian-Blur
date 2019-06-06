/*
* a kernel that executes a gaussian filter
*/
__kernel void gaussian_blur(
	__global const unsigned char* A,
	__global const float* B,
	__global unsigned char* C,
	__local unsigned char* cache,
	int width,
	int height,
	int filterWidth,
	int filterHeight)
{
	size_t global_id = get_global_id(0);
	size_t l_id = get_local_id(0);
	size_t l_size = get_local_size(0);
	size_t g_id = get_group_id(0);
	size_t g_num = get_num_groups(0);

	//printf("Global: %d Local: %d LocalSize: %d GroupId: %d NumGroups: %d\n", global_id, l_id, l_size, g_id, g_num);

	int id = l_id + g_id * l_size;
	int row = id / width;
	int col = id % width;
	int borderWidth = 0;

	// init cache
	if (filterWidth == filterHeight)
	{
		// one-pass
		borderWidth = filterWidth / 2;
		for (unsigned int i = 0; i < filterWidth * filterHeight; i++)
		{
			int posX = filterWidth / 2 - (i % filterWidth);
			int posY = filterHeight / 2 - i / filterHeight;
			//printf("%d: %d|%d\n", id, posX, posY);
			cache[l_id + width * posY + 4 * posX] = A[id + width * posY + 4 * posX];
		}
	}
	else if (filterHeight == 1)
	{
		borderWidth = filterWidth / 2;
		//two-pass horizontal
		for (unsigned int i = 0; i < filterWidth; i++)
		{
			int pos = filterWidth / 2 - i;
			//printf("%d: %d\n", id, pos);
			cache[l_id + 4 * pos] = A[id + 4 * pos];
		}
	}
	else if (filterWidth == 1)
	{
		borderWidth = filterHeight / 2;
		//two-pass vertical
		for (unsigned int i = 0; i < filterHeight; i++)
		{
			int pos = filterHeight / 2 - i;
			//printf("%d: %d\n", id, pos);
			cache[l_id + width * pos] = A[id + width * pos];
		}
	}
	else return;

	barrier(CLK_LOCAL_MEM_FENCE);

	// omit alpha channel
	if (id % 4 == 3)
	{
		C[id] = cache[l_id];
		return;
	}

	// omit border
	if (col < borderWidth * 4 || col >= width - borderWidth * 4 || row < borderWidth || row >= height / 4 - borderWidth)
	{
		C[id] = cache[l_id];
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
			sum += cache[l_id + width * posY + 4 * posX] * B[i];
		}
	}
	else if (filterHeight == 1)
	{
		//two-pass horizontal
		for (unsigned int i = 0; i < filterWidth; i++)
		{
			int pos = filterWidth / 2 - i;
			//printf("%d: %d\n", id, pos);
			sum += cache[l_id + 4 * pos] * B[i];
		}
	}
	else if (filterWidth == 1)
	{
		//two-pass vertical
		for (unsigned int i = 0; i < filterHeight; i++)
		{
			int pos = filterHeight / 2 - i;
			//printf("%d: %d\n", id, pos);
			sum += cache[l_id + width * pos] * B[i];
		}
	}
	else return;

	//printf("Calculated %f\n", sum);

	C[id] = sum;
}
