#include <zmq.hpp>
#include <string>
#include <iostream>
#include <opencv2/opencv.hpp>
#ifdef _WIN32
#include<Windows.h>
#elif defined __unix__
#include <unistd.h>
#endif

void delay(int msec)
{
#ifdef _WIN32
	Sleep(msec);
#elif defined __unix__
	sleep(msec);
#endif
}

int main() {

	void* img_ptr = nullptr;

	int count = 0;
	void* context = zmq_ctx_new();
	//void* publisher = zmq_socket(context, ZMQ_PUB);
	void* publisher = zmq_socket(context, ZMQ_PUSH);
	int bind = zmq_bind(publisher, "tcp://*:5555");


	while (1) {
		// Reading the image through opencv package
		std::cout << count << std::endl;
		cv::Mat image = cv::imread("D:\\서산 이미지\\DP3_3Shot\\DP3_3Shot_원본 50개\\141813\\Virtual BCR\\F 1_1.tif", -1);
		int height = image.rows;
		int width = image.cols;
		zmq_send(publisher, image.data, (height * width * sizeof(UINT16)), ZMQ_NOBLOCK);
		count += 1;
	}
	return 0;

}