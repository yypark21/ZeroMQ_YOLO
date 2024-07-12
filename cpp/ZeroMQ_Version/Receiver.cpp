#include "Receiver.h"

using namespace ZeroMQ;

Receiver::Receiver()
{
	m_pc_ip_ = "tcp://*:5555";
	m_image_address_ = "";
}

Receiver::~Receiver() {}

void Receiver::ReceiveImage()
{

	/*
	cv::Mat image = cv::Mat::zeros (2048, 3072, CV_16U);
	int height, width;
	void* context = zmq_ctx_new();
	void* subscriber = zmq_socket(context, ZMQ_PULL);


	uint16_t* data = nullptr;
	int* num = nullptr;

	data = new uint16_t[2048 * 3072];
	num = new int;

	zmq_bind(subscriber, m_pc_ip_);
	//zmq_recv(publisher, data, (3072 * 2048 * sizeof(UINT16)), ZMQ_NOBLOCK);
	zmq_recv(subscriber, num, (sizeof(int)), ZMQ_NOBLOCK);
*/

	zmq::context_t context(1);
	zmq::socket_t socket(context, ZMQ_REP);
	socket.bind(m_pc_ip_);

	zmq::message_t request;
	zmq_recv(socket, &request, sizeof(char) * 5, ZMQ_NOBLOCK);

	
	int stop = 0;
}

void Receiver::SetPCIP(const char* pc_ip)
{
	m_pc_ip_ = pc_ip;
}

const char* Receiver::GetPCIP()
{
	return m_pc_ip_;
}

void Receiver::SetImageAddress(std::string str_address)
{
	m_image_address_ = str_address;
}

const std::string Receiver::GetImageAddress()
{
	return m_image_address_;
}