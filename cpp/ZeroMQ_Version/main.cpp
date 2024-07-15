#include "main.h"
#include "Sender.h"
#include "Receiver.h"

int main()
{	
	ZeroMQ::Sender sender;

	sender.SetPCIP("tcp://*:5555");
	sender.SetImageAddress("../../image/F 1_1.tif");

	int count = 0;

	
	while (1) {
		// Reading the image through opencv package
		std::cout << count << std::endl;
		sender.SendImage();
		count += 1;
	}
	

	
	ZeroMQ::Receiver receiver;

	while (1) {
		// Reading the image through opencv package
		//std::cout << count << std::endl;
		receiver.ReceiveImage();
		//std::count += 1;
	}
	
	return 0;
}
