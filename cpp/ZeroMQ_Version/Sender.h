#pragma once
#include "main.h"

namespace ZeroMQ
{
	class Sender
	{
	public:
		Sender();
		virtual ~Sender();
		void SendImage();
		void SetPCIP(const char* pc_ip);
		const char* GetPCIP();
		void SetImageAddress(std::string str_address);
		const std::string GetImageAddress();
	private:
		const char* m_pc_ip_ = nullptr;
		std::string m_image_address_;
		cv::Mat m_image_;
		void* m_context_;
		void* m_publisher_;
	};
}
