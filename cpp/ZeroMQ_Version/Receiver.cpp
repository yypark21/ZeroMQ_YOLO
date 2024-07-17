#include "Receiver.h"

using namespace ZeroMQ;

Receiver::Receiver()
{
	m_pc_ip_ = "tcp://localhost:5555";
	m_context_ = zmq::context_t(1);
	m_socket_ = zmq::socket_t(m_context_, ZMQ_PULL);
}

Receiver::~Receiver() {}

void Receiver::ReceiveImage()
{
	zmq::message_t request;
	m_socket_.recv(request);
	cv::Mat image(2048, 3072, CV_16UC1, request.data());
}

void Receiver::SetPCIP(const char* pc_ip)
{
	m_pc_ip_ = pc_ip;
	m_socket_.connect(m_pc_ip_);
}

const char* Receiver::GetPCIP()
{
	return m_pc_ip_;
}
