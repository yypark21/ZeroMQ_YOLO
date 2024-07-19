#include "Sender.h"

using namespace ZeroMQ;

Sender::Sender()
{
	m_pc_ip_ = "tcp://*:5555";
	m_image_address_ = "";
	m_context_ = zmq_ctx_new();
	m_publisher_ = zmq_socket(m_context_, ZMQ_PUSH);
}

Sender::~Sender(){}

void Sender::SendImage()
{
	std::cout << "sender start \n";
	//zmq_send(m_publisher_, m_image_.data, (m_image_.rows * m_image_.cols * sizeof(UINT16)), ZMQ_NOBLOCK);

	//const char* reply_message = "Hi from ZeroMQ C++ Server";
	zmq_send(m_publisher_, &m_image_.rows, sizeof(int), ZMQ_SNDMORE);
	zmq_send(m_publisher_, &m_image_.cols, sizeof(int), ZMQ_SNDMORE);
	zmq_send(m_publisher_, m_image_.data, (m_image_.rows * m_image_.cols * sizeof(UINT16)), ZMQ_NOBLOCK);
}

void Sender::SetPCIP(const char* pc_ip)
{
	m_pc_ip_ = pc_ip;	
	zmq_bind(m_publisher_, m_pc_ip_);
}

const char* Sender::GetPCIP()
{
	return m_pc_ip_;
}

void Sender::SetImageAddress(const std::string& str_address)
{
	m_image_address_ = str_address;
	m_image_ = cv::imread(m_image_address_, -1);
}

void Sender::SetImage(const cv::Mat& image)
{
	m_image_ = image;
}

cv::Mat& Sender::GetImage()
{
	return m_image_;
}

const std::string Sender::GetImageAddress()
{
	return m_image_address_;
}