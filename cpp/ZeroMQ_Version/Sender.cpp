#include "Sender.h"

using namespace ZeroMQ;

Sender::Sender()
{
	m_pc_ip_ = "tcp://*:5555";
	m_image_address_ = "";
	m_context_ = nullptr;
	m_publisher_ = nullptr;
}

Sender::~Sender(){}

void Sender::SendImage()
{
	m_image_ = cv::imread(m_image_address_, -1);
	zmq_send(m_publisher_, m_image_.data, (m_image_.rows * m_image_.cols * sizeof(UINT16)), ZMQ_NOBLOCK);
}

void Sender::SetPCIP(const char* pc_ip)
{
	m_pc_ip_ = pc_ip;
	m_context_ = zmq_ctx_new();
	m_publisher_ = zmq_socket(m_context_, ZMQ_PUSH);
	zmq_bind(m_publisher_, m_pc_ip_);
}

const char* Sender::GetPCIP()
{
	return m_pc_ip_;
}

void Sender::SetImageAddress(std::string str_address)
{
	m_image_address_ = str_address;
}

const std::string Sender::GetImageAddress()
{
	return m_image_address_;
}