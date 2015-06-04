#ifndef INITIALIZE_H
#define INITIALIZE_H

#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>

#include <iostream>
#include <stdio.h>
#include <sstream>

#include <string>

bool initializeGame(cv::Mat &game_frame, cv::Rect &eyeROI, double* gazeCalibrations);

#endif // INITIALIZE_H
