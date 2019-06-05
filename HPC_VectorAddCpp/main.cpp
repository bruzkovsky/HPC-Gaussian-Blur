﻿#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

#include "CL/cl.h"
#include "tga.h"
#include <malloc.h>
#include <fstream>
#include <string>

std::string cl_errorstring(cl_int err)
{
	switch (err)
	{
	case CL_SUCCESS:									return std::string("Success");
	case CL_DEVICE_NOT_FOUND:							return std::string("Device not found");
	case CL_DEVICE_NOT_AVAILABLE:						return std::string("Device not available");
	case CL_COMPILER_NOT_AVAILABLE:						return std::string("Compiler not available");
	case CL_MEM_OBJECT_ALLOCATION_FAILURE:				return std::string("Memory object allocation failure");
	case CL_OUT_OF_RESOURCES:							return std::string("Out of resources");
	case CL_OUT_OF_HOST_MEMORY:							return std::string("Out of host memory");
	case CL_PROFILING_INFO_NOT_AVAILABLE:				return std::string("Profiling information not available");
	case CL_MEM_COPY_OVERLAP:							return std::string("Memory copy overlap");
	case CL_IMAGE_FORMAT_MISMATCH:						return std::string("Image format mismatch");
	case CL_IMAGE_FORMAT_NOT_SUPPORTED:					return std::string("Image format not supported");
	case CL_BUILD_PROGRAM_FAILURE:						return std::string("Program build failure");
	case CL_MAP_FAILURE:								return std::string("Map failure");
	case CL_MISALIGNED_SUB_BUFFER_OFFSET:				return std::string("Misaligned sub buffer offset");
	case CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST:	return std::string("Exec status error for events in wait list");
	case CL_INVALID_VALUE:                    			return std::string("Invalid value");
	case CL_INVALID_DEVICE_TYPE:              			return std::string("Invalid device type");
	case CL_INVALID_PLATFORM:                 			return std::string("Invalid platform");
	case CL_INVALID_DEVICE:                   			return std::string("Invalid device");
	case CL_INVALID_CONTEXT:                  			return std::string("Invalid context");
	case CL_INVALID_QUEUE_PROPERTIES:         			return std::string("Invalid queue properties");
	case CL_INVALID_COMMAND_QUEUE:            			return std::string("Invalid command queue");
	case CL_INVALID_HOST_PTR:                 			return std::string("Invalid host pointer");
	case CL_INVALID_MEM_OBJECT:               			return std::string("Invalid memory object");
	case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:  			return std::string("Invalid image format descriptor");
	case CL_INVALID_IMAGE_SIZE:               			return std::string("Invalid image size");
	case CL_INVALID_SAMPLER:                  			return std::string("Invalid sampler");
	case CL_INVALID_BINARY:                   			return std::string("Invalid binary");
	case CL_INVALID_BUILD_OPTIONS:            			return std::string("Invalid build options");
	case CL_INVALID_PROGRAM:                  			return std::string("Invalid program");
	case CL_INVALID_PROGRAM_EXECUTABLE:       			return std::string("Invalid program executable");
	case CL_INVALID_KERNEL_NAME:              			return std::string("Invalid kernel name");
	case CL_INVALID_KERNEL_DEFINITION:        			return std::string("Invalid kernel definition");
	case CL_INVALID_KERNEL:                   			return std::string("Invalid kernel");
	case CL_INVALID_ARG_INDEX:                			return std::string("Invalid argument index");
	case CL_INVALID_ARG_VALUE:                			return std::string("Invalid argument value");
	case CL_INVALID_ARG_SIZE:                 			return std::string("Invalid argument size");
	case CL_INVALID_KERNEL_ARGS:             			return std::string("Invalid kernel arguments");
	case CL_INVALID_WORK_DIMENSION:          			return std::string("Invalid work dimension");
	case CL_INVALID_WORK_GROUP_SIZE:          			return std::string("Invalid work group size");
	case CL_INVALID_WORK_ITEM_SIZE:           			return std::string("Invalid work item size");
	case CL_INVALID_GLOBAL_OFFSET:            			return std::string("Invalid global offset");
	case CL_INVALID_EVENT_WAIT_LIST:          			return std::string("Invalid event wait list");
	case CL_INVALID_EVENT:                    			return std::string("Invalid event");
	case CL_INVALID_OPERATION:                			return std::string("Invalid operation");
	case CL_INVALID_GL_OBJECT:                			return std::string("Invalid OpenGL object");
	case CL_INVALID_BUFFER_SIZE:              			return std::string("Invalid buffer size");
	case CL_INVALID_MIP_LEVEL:                			return std::string("Invalid mip-map level");
	case CL_INVALID_GLOBAL_WORK_SIZE:         			return std::string("Invalid gloal work size");
	case CL_INVALID_PROPERTY:                 			return std::string("Invalid property");
	default:                                  			return std::string("Unknown error code");
	}
}

void checkStatus(cl_int err)
{
	if (err != CL_SUCCESS) {
		printf("OpenCL Error: %s \n", cl_errorstring(err).c_str());
		exit(EXIT_FAILURE);
	}
}

void printCompilerError(cl_program program, cl_device_id device)
{
	cl_int status;
	size_t logSize;
	char* log;

	// get log size
	status = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &logSize);
	checkStatus(status);

	// allocate space for log
	log = static_cast<char*>(malloc(logSize));
	if (!log)
	{
		exit(EXIT_FAILURE);
	}

	// read the log
	status = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, logSize, log, NULL);
	checkStatus(status);

	// print the log
	printf("Build Error: %s\n", log);
}

void printVector(int32_t* vector, unsigned int elementSize, const char* label)
{
	printf("%s:\n", label);

	for (unsigned int i = 0; i < elementSize; ++i)
	{
		printf("%d ", vector[i]);
	}

	printf("\n");
}

void printCharVector(unsigned char* vector, unsigned int elementSize, const char* label)
{
	printf("%s:\n", label);

	for (unsigned int i = 0; i < elementSize; ++i)
	{
		printf("%d ", vector[i]);
	}

	printf("\n");
}

int main(int argc, char** argv)
{
	tga::TGAImage image;
	bool loaded = tga::LoadTGA(&image, "image.tga");
	if (!loaded)
	{
		printf("Error: TGA file could not be loaded.\n");
		exit(EXIT_FAILURE);
	}

	size_t imageSize = image.imageData.size();
	unsigned char* imageData = image.imageData.data();

	cl_int imageWidth = image.width * 4;
	cl_int imageHeight = image.height * 4;

	// output result
	//printCharVector(image.imageData.data(), imageSize, "Input image");

	// input and output arrays
	cl_int filterWidth = 3;
	cl_int filterHeight = 3;
	size_t dataSize = filterWidth * filterHeight * sizeof(float);
	//int32_t* vectorA = static_cast<int32_t*>(malloc(dataSize));
	unsigned char* vectorC = static_cast<unsigned char *>(malloc(imageSize));

	//for (unsigned int i = 0; i < filterSize; ++i)
	//{
	//	vectorA[i] = static_cast<int32_t>(i);
	//}

	// data from http://dev.theomader.com/gaussian-kernel-calculator/ with sigma 1 and size 3
	float vectorB[] = {
		0.077847,	0.123317,	0.077847,
		0.123317,	0.195346,	0.123317,
		0.077847,	0.123317,	0.077847
	};

	// used for checking error status of api calls
	cl_int status;

	// retrieve the number of platforms
	cl_uint numPlatforms = 0;
	clGetPlatformIDs(0, NULL, &numPlatforms);

	if (numPlatforms == 0)
	{
		printf("Error: No OpenCL platform available!\n");
		exit(EXIT_FAILURE);
	}

	// select the platform
	cl_platform_id platform;
	checkStatus(clGetPlatformIDs(1, &platform, NULL));

	// retrieve the number of devices
	cl_uint numDevices = 0;
	checkStatus(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, NULL, &numDevices));

	if (numDevices == 0)
	{
		printf("Error: No OpenCL device available for platform!\n");
		exit(EXIT_FAILURE);
	}

	// select the device
	cl_device_id device;
	checkStatus(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 1, &device, NULL));

	// create context
	cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &status);
	checkStatus(status);

	// create command queue
	cl_command_queue commandQueue = clCreateCommandQueue(context, device, 0, &status);
	checkStatus(status);

	// allocate two input and one output buffer for the three vectors
	cl_mem bufferA = clCreateBuffer(context, CL_MEM_READ_ONLY, imageSize, NULL, &status);
	checkStatus(status);
	cl_mem bufferB = clCreateBuffer(context, CL_MEM_READ_ONLY, dataSize, NULL, &status);
	checkStatus(status);
	cl_mem bufferC = clCreateBuffer(context, CL_MEM_WRITE_ONLY, imageSize, NULL, &status);
	checkStatus(status);

	// write data from the input vectors to the buffers
	checkStatus(clEnqueueWriteBuffer(commandQueue, bufferA, CL_TRUE, 0, imageSize, imageData, 0, NULL, NULL));
	checkStatus(clEnqueueWriteBuffer(commandQueue, bufferB, CL_TRUE, 0, dataSize, vectorB, 0, NULL, NULL));

	// read the kernel source
	const char* kernelFileName = "kernel.cl";
	std::ifstream ifs(kernelFileName);
	if (!ifs.good())
	{
		printf("Error: Could not open kernel with file name %s!\n", kernelFileName);
		exit(EXIT_FAILURE);
	}

	std::string programSource((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
	const char* programSourceArray = programSource.c_str();
	size_t programSize = programSource.length();

	// create the program
	cl_program program = clCreateProgramWithSource(context, 1, static_cast<const char**>(&programSourceArray), &programSize, &status);
	checkStatus(status);

	// build the program
	status = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
	if (status != CL_SUCCESS)
	{
		printCompilerError(program, device);
		exit(EXIT_FAILURE);
	}

	// create the gaussian blur kernel
	cl_kernel kernel = clCreateKernel(program, "gaussian_blur", &status);
	checkStatus(status);

	// set the kernel arguments
	checkStatus(clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufferA));
	checkStatus(clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufferB));
	checkStatus(clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufferC));
	checkStatus(clSetKernelArg(kernel, 3, sizeof(cl_int), &imageWidth));
	checkStatus(clSetKernelArg(kernel, 4, sizeof(cl_int), &imageHeight));
	checkStatus(clSetKernelArg(kernel, 5, sizeof(cl_int), &filterWidth));
	checkStatus(clSetKernelArg(kernel, 6, sizeof(cl_int), &filterHeight));

	// define an index space of work-items for execution
	cl_uint maxWorkItemDimensions;
	checkStatus(clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(cl_uint), &maxWorkItemDimensions, NULL));

	size_t* maxWorkItemSizes = static_cast<size_t*>(malloc(maxWorkItemDimensions * sizeof(size_t)));
	checkStatus(clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_SIZES, maxWorkItemDimensions * sizeof(size_t), maxWorkItemSizes, NULL));

	size_t globalWorkSize;
	if (imageWidth > maxWorkItemSizes[0])
		globalWorkSize = static_cast<size_t>(maxWorkItemSizes[0]);
	else
		globalWorkSize = static_cast<size_t>(imageWidth);

	printf("Max work item size: %d; using %d\n", maxWorkItemSizes[0], globalWorkSize);

	free(maxWorkItemSizes);

	size_t localWorkSize = 64;

	printf("Local work size: %d\n", localWorkSize);

	printf("Executing kernel for image width/height %d/%d\n", imageWidth, imageHeight);

	// execute the kernel
	for (unsigned int i = 0; i < imageSize; i += globalWorkSize)
	{
		size_t offset = static_cast<size_t>(i);
		checkStatus(clEnqueueNDRangeKernel(commandQueue, kernel, 1, &offset, &globalWorkSize, &localWorkSize, 0, NULL, NULL));
	}

	// read the device output buffer to the host output array
	checkStatus(clEnqueueReadBuffer(commandQueue, bufferC, CL_TRUE, 0, imageSize, vectorC, 0, NULL, NULL));

	// CPP implementation
	//
	//for (unsigned int i = 0; i < imageSize; i++)
	//{
	//	int row = i / imageWidth;
	//	int col = i % imageWidth;

	//	if (i % 4 == 3)
	//	{
	//		vectorC[i] = 255;
	//		continue;
	//	}

	//	if (col < 4 || col >= imageWidth - 4 || row < 4 || row >= imageHeight - 4)
	//		continue;

	//	float sum = 0.0f;
	//	sum += imageData[(i - imageWidth) - 1] * vectorB[8];
	//	sum += imageData[i - imageWidth] * vectorB[7];
	//	sum += imageData[(i - imageWidth) + 1] * vectorB[6];
	//	sum += imageData[i - 1] * vectorB[5];
	//	sum += imageData[i] * vectorB[4];
	//	sum += imageData[i + 1] * vectorB[3];
	//	sum += imageData[i + imageWidth - 1] * vectorB[2];
	//	sum += imageData[i + imageWidth] * vectorB[1];
	//	sum += imageData[i + imageWidth + 1] * vectorB[0];

	//	//printf("data: %d; sum: %f\n", imageData[i], sum);

	//	vectorC[i] = sum;
	//}

	//printVector(vectorB, elementSize, "Input B");
	//printCharVector(vectorC, imageSize, "Output C");

	tga::TGAImage resultImage;

	resultImage.bpp = image.bpp;
	resultImage.width = image.width;
	resultImage.height = image.height;
	resultImage.type = image.type;

	resultImage.imageData = std::vector<unsigned char>(imageSize);

	for (unsigned int i = 0; i < imageSize; i++)
		resultImage.imageData[i] = vectorC[i];

	tga::saveTGA(resultImage, "output.tga");

	/*int currentByte = 0;

	for (int py = 0; py < resultImage.height; ++py)
		for (int px = 0; px < resultImage.width; ++px)
		{
			resultImage.imageData[currentByte++] = image.imageData[currentByte];
			resultImage.imageData[currentByte++] = image.imageData[currentByte];
			resultImage.imageData[currentByte++] = image.imageData[currentByte];
		}*/

	// release allocated resources
	free(vectorC);
	//free(vectorB);
	//free(vectorA);
	//free(imageData);

	// release opencl objects
	checkStatus(clReleaseKernel(kernel));
	checkStatus(clReleaseProgram(program));
	checkStatus(clReleaseMemObject(bufferC));
	checkStatus(clReleaseMemObject(bufferB));
	checkStatus(clReleaseMemObject(bufferA));
	checkStatus(clReleaseCommandQueue(commandQueue));
	checkStatus(clReleaseContext(context));

	exit(EXIT_SUCCESS);
}
