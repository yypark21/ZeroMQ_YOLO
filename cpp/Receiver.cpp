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
	cv::Mat image;
	int height, width;
	void* context = zmq_ctx_new();
	void* publisher = zmq_socket(context, ZMQ_PUSH);

	zmq_bind(publisher, m_pc_ip_);
	image = cv::imread(m_image_address_, -1);
	zmq_send(publisher, image.data, (image.rows * image.cols * sizeof(UINT16)), ZMQ_NOBLOCK);
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