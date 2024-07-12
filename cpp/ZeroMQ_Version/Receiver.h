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
		std::string m_image_address_;
	};
}
