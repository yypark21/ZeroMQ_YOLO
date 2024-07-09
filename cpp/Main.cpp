#include "main.h"
#include "Sender.h"

int main()
{
	ZeroMQ::Sender sender;
	sender.SetPCIP("tcp://*:5555");
	sender.SetImageAddress("D:\\서산 이미지\\DP3_3Shot\\DP3_3Shot_원본 50개\\141813\\Virtual BCR\\F 1_1.tif");

	//void* publisher = zmq_socket(context, ZMQ_PUB);

	int count = 0;

	while (1) {
		// Reading the image through opencv package
		std::cout << count << std::endl;
		sender.SendImage();
		count += 1;
	}

	return 0;
}