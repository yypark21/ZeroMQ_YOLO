#include "main.h"
#include "Sender.h"
#include "Receiver.h"

int main()
{	
	ZeroMQ::Sender sender;
	sender.SetPCIP("tcp://*:5555");
	sender.SetImageAddress("../../image/F 1_1.tif");

	ZeroMQ::Receiver receiver;
	receiver.SetPCIP("tcp://localhost:5555");

	std::future<void> f1, f2;

	int count = 0;
	
	while (1) {
		std::cout << count << std::endl;

		//sender.SendImage();
		receiver.ReceiveImage();

		//f1 = std::async(&ZeroMQ::Sender::SendImage, sender);
		//f2 = std::async(&ZeroMQ::Receiver::ReceiveImage, receiver);

		count += 1;
	}


	//f1.get();
	//f2.get();
	
	return 0;
}
