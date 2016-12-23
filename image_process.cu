#include <tiffio.h>
#include <chrono>
#include <string>
#include <algorithm>
#include <map>
#include <iostream>
#include <sstream>
#include <cufft.h>
#include <cuComplex.h>
#include <cuda_runtime.h>
#include <thrust/complex.h>
#include <thrust/extrema.h>
#include <thrust/device_vector.h> 
#include <thrust/transform.h>
#include <thrust/sequence.h>
#include <thrust/copy.h> 
#include <thrust/fill.h>
#include <thrust/replace.h> 
#include <thrust/functional.h>



using namespace thrust;
using namespace std;


typedef struct{
	thrust::host_vector<unsigned short> img;
	int height;
	int width;
	unsigned short bits_per_sample;
}Image;



#define CudaSafeCall( err ) __CudaSafeCall(err, __FILE__, __LINE__ )
inline void __CudaSafeCall(cudaError err, const char *file, const int line)
{
	if (cudaSuccess != err)
	{
		std::stringstream error_msg;
		error_msg << "CudaSafeCall() failed at " << file << ":" << line << ":" << cudaGetErrorString(err);
		auto error_msg_str = error_msg.str();
		std::cout << error_msg_str << std::endl;
		throw std::runtime_error(error_msg_str);
	}
}

#define CufftSafeCall( err ) __CufftSafeCallCall( err, __FILE__, __LINE__ )
static const char *_cufftGetErrorEnum(cufftResult_t error)
{
	switch (error)
	{
	case CUFFT_SUCCESS:
		return "CUFFT_SUCCESS";
	case CUFFT_INVALID_PLAN:
		return "CUFFT_INVALID_PLAN";
	case CUFFT_ALLOC_FAILED:
		return "CUFFT_ALLOC_FAILED";
	case CUFFT_INVALID_TYPE:
		return "CUFFT_INVALID_TYPE";
	case CUFFT_INVALID_VALUE:
		return "CUFFT_INVALID_VALUE";
	case CUFFT_INTERNAL_ERROR:
		return "CUFFT_INTERNAL_ERROR";
	case CUFFT_EXEC_FAILED:
		return "CUFFT_EXEC_FAILED";
	case CUFFT_SETUP_FAILED:
		return "CUFFT_SETUP_FAILED";
	case CUFFT_INVALID_SIZE:
		return "CUFFT_INVALID_SIZE";
	case CUFFT_UNALIGNED_DATA:
		return "CUFFT_UNALIGNED_DATA";
	case CUFFT_INCOMPLETE_PARAMETER_LIST: break;
	case CUFFT_INVALID_DEVICE: break;
	case CUFFT_PARSE_ERROR: break;
	case CUFFT_NO_WORKSPACE: break;
	case CUFFT_NOT_IMPLEMENTED: break;
	case CUFFT_LICENSE_ERROR: break;
	default: break;
	}
	return "<unknown>";
}

inline void __CufftSafeCallCall(cufftResult_t err, const char *file, const int line)
{
	if (CUFFT_SUCCESS != err)
	{
		std::stringstream error_msg;
		error_msg << "CufftSafeCall() failed at " << file << ":" << line << ":" << _cufftGetErrorEnum(err);
		auto error_msg_str = error_msg.str();
		std::cout << error_msg_str << std::endl;
		throw std::runtime_error(error_msg_str);
	}
}






static Image ReadBuffer(const std::string& name)
{
	auto tif = TIFFOpen(name.c_str(), "r");
	if (tif == nullptr)
	{
		std::stringstream error_msg;
		error_msg << "Can't Open File " << name;
		auto error_msg_str = error_msg.str();
		std::cout << error_msg_str << std::endl;
		throw std::runtime_error(error_msg_str);
	}
	int imgH, imgW;
	TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &imgH);
	TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &imgW);
	unsigned short bits_per_sample;
	TIFFGetField(tif, TIFFTAG_BITSPERSAMPLE, &bits_per_sample);
	if (bits_per_sample != sizeof(unsigned short) * 8)
	{
		std::stringstream error_msg;
		error_msg << "File: " + std::string(name) + " has an unssuported bit depth of " + std::to_string(bits_per_sample);
		auto error_msg_str = error_msg.str();
		std::cout << error_msg_str << std::endl;
		throw std::runtime_error(error_msg_str);
	}
	Image ret;
	ret.height = imgH;
	ret.width = imgW;
	ret.bits_per_sample = bits_per_sample;
	int rowsize = imgW*sizeof(unsigned short);
	ret.img.resize(imgW*imgH);
	// static_cast<FrameSize&>(ret) = FrameSize(imgW, imgH);
	auto mydata = reinterpret_cast<unsigned char*>(ret.img.data());
	for (auto row = 0; row < imgH; row++)
	{
		auto toMe = static_cast<void*>(&mydata[rowsize*row]);
		TIFFReadScanline(tif, toMe, row);
	}
	TIFFClose(tif);
	return ret;
}


template<typename T>
void WriteDebugCuda(const T* pointer_in, const size_t width, const size_t height, const char* name, bool do_it_anyways = false)
{
	if (do_it_anyways || debug_writes)
	{
		//todo replace with thrust?
		auto bytes = width*height*sizeof(T);
		T* temp = nullptr;
		CudaSafeCall(cudaDeviceSynchronize());
		CudaSafeCall(cudaHostAlloc((void**)&temp, bytes, cudaHostAllocDefault));
		CudaSafeCall(cudaMemcpy(temp, pointer_in, bytes, cudaMemcpyDeviceToHost));
		WriteTiff(name, temp, width, height);
		CudaSafeCall(cudaFreeHost(temp));
	}
}

struct saxpy_functor {  
	saxpy_functor(){} 
	__host__ __device__ 
		thrust::complex<float> operator()(const thrust::complex<float>& x, const thrust::complex<float>& y) const {
		return x*y / abs(x*y); 
	} 
};

struct conjugate_functor {
	conjugate_functor(){}
	__host__ __device__
		thrust::complex<float> operator()(const thrust::complex<float>& x) const {
		return 2*x.real() - x;
	}
};

struct Scaling_functor {
	const int size;
	Scaling_functor(int _size) : size(_size){}
	__host__ __device__
		float operator()(const float& x) const {
		return x/size;
	}
};
/*
Description: Get the translation offset between two images
Input: Reference Image input1, Image(to be shifted) input2, row numbers, col numbers
Output: None
*/
void phase_correlation(device_vector<float>& input1, device_vector<float>& input2, cufftHandle& plan1,int row, int col, float& x_shift, float& y_shift)
{
	int position, x_position, y_position;
	float max_value, delta_x1, delta_y1;
	cufftHandle plan2;
	device_vector<thrust::complex<float>> temp1(row*col);
	device_vector<thrust::complex<float>>temp2(row*col);
	device_vector<float> temp3(row*col);
	
	
	auto temp1_ptr = (cufftComplex*)raw_pointer_cast(temp1.data());
	auto temp2_ptr = (cufftComplex*)raw_pointer_cast(temp2.data());
	auto temp3_ptr = raw_pointer_cast(temp3.data());
	auto input1_ptr = raw_pointer_cast(input1.data());
	auto input2_ptr = raw_pointer_cast(input2.data());
	
	//fft
	CufftSafeCall(cufftExecR2C(plan1, input1_ptr, temp1_ptr));
	CufftSafeCall(cufftExecR2C(plan1, input2_ptr, temp2_ptr));
	
	if (plan1)
	   cufftDestroy(plan1);
	
	
	// get the conjugate
	thrust::transform(temp2.begin(), temp2.end(),temp2.begin(), conjugate_functor());
	thrust::transform(temp1.begin(), temp1.end(), temp2.begin(), temp2.begin(), saxpy_functor());

	
	//ifft
	cufftPlan2d(&plan2, row, col, CUFFT_C2R);
	CufftSafeCall(cufftExecC2R(plan2, temp2_ptr, temp3_ptr));
	cufftDestroy(plan2);
	//Scailing back after un-normalized fft

	thrust::transform(temp3.begin(), temp3.end(), temp3.begin(), Scaling_functor(row*col));


	//Find the peak
	thrust::device_vector<float>::iterator iter = thrust::max_element(temp3.begin(), temp3.end());
	position = iter - temp3.begin();
	max_value = *iter;
	y_position = position / col;
	x_position = position%col;
	delta_x1 = (temp3[position + 1] * (x_position + 1) + temp3[position] * x_position) / (temp3[position + 1] + temp3[position]);
	delta_y1 = (temp3[position + col] * (y_position + 1) + temp3[position] * y_position) / (temp3[position + col] + temp3[position]);


	if (delta_x1 < col / 2)
		x_shift = delta_x1;
	else
		x_shift = delta_x1 - row;
	if (delta_y1 < row / 2)
		y_shift = delta_y1;
	else
		y_shift = delta_y1 - col;
	/*
	cout << "x_position: " << x_position << endl;
	cout << "y_position: " << y_position << endl;
	cout << "delta_x1: " << delta_x1 << endl;
	cout << "delta_y1: " << delta_y1 << endl;
	cout << "x_shift: " << x_shift << endl;
	cout << "y_shift: " << y_shift << endl;
	*/
	
}

int main()
{
	Image a,b;
	float x, y;
	cufftHandle plan;
	
	chrono::time_point<std::chrono::system_clock> start, end;
	/* test examples*/
	string name1 = "E:/research/test_case/f0_t0_i0_ch1_c0_r0_z1_m0.tif";
	string name2 = "E:/research/test_case/f0_t0_i0_ch1_c0_r0_z1_m2.tif";
	a = ReadBuffer(name1);
	b = ReadBuffer(name2);
	device_vector<float> image1,image2;
	auto&  input1 = a.img;
	auto&  input2 = b.img;
	image1.resize(input1.size());
	image2.resize(input2.size());
	thrust::copy(input1.begin(), input1.end(), image1.begin());
	thrust::copy(input2.begin(), input2.end(), image2.begin());
	cufftPlan2d(&plan, a.height, a.width, CUFFT_R2C);
	start = chrono::system_clock::now();
	phase_correlation(image1, image2, plan, a.height, a.width, x, y);
	end = chrono::system_clock::now();
	chrono::duration<double> elapsed_seconds = end - start;
	cout << x << ";" << y << endl;
	cout << "Cycle Time: " << elapsed_seconds.count() << "s" << std::endl;
	return 0;
}

//-0.092,-0.067