#pragma once
#include "main.h"

namespace ZeroMQ
{
	class Receiver
	{
	public:
		Receiver();
		virtual ~Receiver();
		void ReceiveImage();
		void SetPCIP(const char* pc_ip);
		const char* GetPCIP();
		void SetImageAddress(std::string str_address);
		const std::string GetImageAddress();
	private:
		const char* m_pc_ip_ = nullptr;
		zmq::socket_t m_socket_;
		zmq::context_t m_context_;
	};
}
