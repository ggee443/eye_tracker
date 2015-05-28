/**
 * @file objectDetection2.cpp
 * @author A. Huaman ( based in the classic facedetect.cpp in samples/c )
 * @brief A simplified version of facedetect.cpp, show how to load a cascade classifier and how to find objects (Face + eyes) in a video stream - Using LBP here
 */
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>

#include <iostream>
#include <stdio.h>
#include <sstream>

#include "findEyeCenter.h"
#include "findEyeCorner.h"


using namespace std;
using namespace cv;

/** Function Headers */
int detectAndDisplay( Mat frame );

/** Global variables */
String face_cascade_name = "lbpcascade_frontalface.xml";
String eyes_cascade_name = "haarcascade_eye_tree_eyeglasses.xml";
String cascade_path = "/home/ee443/FinalProjectEE443/eye_tracker/src/face_detect/";

CascadeClassifier face_cascade;
CascadeClassifier eyes_cascade;
string window_name = "Capture - Face detection";

RNG rng(12345);

// Initialize Mem
vector<Rect> faces;
Mat frame_gray;
Mat faceROI;
vector<Rect> eyes;
Rect eye;
Point crosshair1;
Point crosshair2;

/**
 * @function main
 */
int main( void )
{
  VideoCapture capture(0);  //sudo apt-get install v4l2ucp v4l-utils libv4l-dev
  Mat frame;
  int cc;

  // Timing Tests
  clock_t start;
  double dur;
  double avg_dur = 750.;
  int num_faces;

  //-- 1. Load the cascade
  if( !face_cascade.load( cascade_path + face_cascade_name ) ){ printf("--(!)objectDetection2:Error loading face_cascade files\n"); return -1; };
  if( !eyes_cascade.load( cascade_path + eyes_cascade_name ) ){ printf("--(!)objectDetection2:Error loading eyes_cascade files\n"); return -1; };

  //-- 2. Read the video stream
  if( capture.isOpened() )
  {      
    while((char)cc != 'q')
    {
      //-- 3. Apply the classifier to the frame
      if( capture.read(frame) ) {
            start = clock();

            flip(frame, frame, 1);
            num_faces = detectAndDisplay( frame );
//--(!) testing
//            imshow("Frame", frame);

            if(num_faces > 0) {
                dur = (clock()-start) * 1000. / (double)CLOCKS_PER_SEC / (double) num_faces;
                avg_dur = 0.9*avg_dur + 0.1*dur;
                printf("Duration: %3.0fms\n", avg_dur);
            }
        }
      else
      { printf(" --(!) No captured frame -- Break!\n"); break; }

      cc = waitKey(1);

    }
  }
  return 0;
}

/**
 * @function detectAndDisplay
 */


//--(!) figure out how to make this faster!
// a: nuke current ubuntu 12.04 and install ubuntu 12.04LTS
// b: only pass mats by reference
int detectAndDisplay( Mat frame )
{
   int numFaces = 0;

   cvtColor( frame, frame_gray, COLOR_BGR2GRAY );

   equalizeHist( frame_gray, frame_gray ); // maybe remove this for speed if unnecessary for quality

   //-- Detect faces
   face_cascade.detectMultiScale( frame_gray, faces, 1.1, 2, 0, Size(80, 80) );

   for( size_t i = 0; i < faces.size(); i++ )
    {
      faceROI = frame_gray( faces[i] );

      //-- In each face, detect eyes
      eyes_cascade.detectMultiScale( faceROI, eyes, 1.1, 2, 0 |CV_HAAR_SCALE_IMAGE, Size(30, 30) );
      if( eyes.size() == 2)
      {
         //-- Draw the face
         rectangle( frame, faces[i],  Scalar( 255, 0, 0 ), 2, 8, 0 );
         numFaces++;

         for( size_t j = 0; j < eyes.size(); j++ )
          { //-- Draw the eyes
            eye.x = eyes[j].x + faces[i].x;
            eye.y = eyes[j].y + faces[i].y;
            eye.width = eyes[j].width;
            eye.height = eyes[j].height;
            rectangle( frame, eye, Scalar( 255, 0, 255 ), 2, 8, 0 );

            // Find Eye Center
//            Point eye_center( eyes[j].width/2, eyes[j].height/2 );  // middle of eye rectangle
            Point eye_center = findEyeCenter(faceROI, eyes[j], "Debug Window");

            // Shift relative to eye
            eye_center.x += faces[i].x + eyes[j].x;
            eye_center.y += faces[i].y + eyes[j].y;

            // Draw Eye Center
            crosshair1.x = eye_center.x;
            crosshair1.y = eye_center.y-5;
            crosshair2.x = eye_center.x;
            crosshair2.y = eye_center.y+5;
            line( frame, crosshair1, crosshair2, Scalar( 0, 255, 0 ), 1, 8, 0 );
            crosshair1.x = eye_center.x-5;
            crosshair1.y = eye_center.y;
            crosshair2.x = eye_center.x+5;
            crosshair2.y = eye_center.y;
            line( frame, crosshair1, crosshair2, Scalar( 0, 255, 0 ), 1, 8, 0 );
          }
       }
    }
   //-- Show what you got
   imshow( window_name, frame );

   return numFaces;
}

