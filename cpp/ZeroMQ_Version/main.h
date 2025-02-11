#pragma once
#include <zmq.hpp>
#include <string>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <thread>
#include <future>
#include <fstream>
#include <chrono>

#ifdef _WIN32
#include<Windows.h>
#elif defined __unix__
#include <unistd.h>
#endif
