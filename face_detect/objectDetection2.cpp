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

//#define GAME_PLAY
#define DEBUG

using namespace std;
using namespace cv;

/** Function Headers */
int detectAndDisGAME_PLAY( Mat &frame );
Point getGaze(int x, int y);
void waitForSpace();

/** Global variables */
String face_cascade_name = "lbpcascade_frontalface.xml";
String eyes_cascade_name = "haarcascade_eye_tree_eyeglasses.xml";
String cascade_path = "/home/ee443/FinalProjectEE443/eye_tracker/src/face_detect/";

CascadeClassifier face_cascade;
CascadeClassifier eyes_cascade;
string window_name = "Capture - Face detection";
string game_window = "Eye Tracking Game";

RNG rng(12345);

// Initialize Mem
vector<Rect> faces;
Mat frame_gray;
Mat faceROI;
vector<Rect> eyes;
Rect eye;
Point crosshair1, crosshair2;
int cc;

// Game
int width_screen = 1590;
int height_screen = 1100;
Mat game_frame;
//double x_frac, y_frac;
double x_frac = 0.5;
double y_frac = 0.9;
Point gaze;

// TopLeft, TopRight, BottomLeft, BottomRight, Middle
Point gazeCalibrations[5];



// Timing Tests
clock_t start;
double dur;
double avg_dur = 750.;
int num_faces;


/**
 * @function main
 */
int main( void )
{
  VideoCapture capture(0);  //sudo apt-get install v4l2ucp v4l-utils libv4l-dev
  Mat frame;

  //-- 1. Load the cascade
  if( !face_cascade.load( cascade_path + face_cascade_name ) ){ printf("--(!)objectDetection2:Error loading face_cascade files\n"); return -1; };
  if( !eyes_cascade.load( cascade_path + eyes_cascade_name ) ){ printf("--(!)objectDetection2:Error loading eyes_cascade files\n"); return -1; };


#ifdef GAME_PLAY
  // Game interface
  // Perform calibration
  game_frame = Mat::zeros(height_screen, width_screen, CV_8UC3);
  imshow( game_window, game_frame ); waitKey(1);
  disGAME_PLAYOverlay(game_window, "Welcome to the game!\n Press 'space' to continue.", -1);
  waitForSpace();

  disGAME_PLAYOverlay(game_window, "Great! Next we are going to do some calibrations.  So get ready to stare at the circle for a bit.\n  Again, press 'space' to continue.", -1);
  waitForSpace();

  gazeCalibrations[0] = getGaze(20,60);
  gazeCalibrations[1] = getGaze(width_screen-20,60);
  gazeCalibrations[2] = getGaze(20,height_screen-20);
  gazeCalibrations[3] = getGaze(width_screen-20,height_screen-20);
  gazeCalibrations[4] = getGaze(width_screen/2,height_screen/2);



//  game_frame = Mat::zeros(height_screen, width_screen, CV_8UC3); // refresh?
//  imshow( game_window, game_frame ); waitKey(1);
  disGAME_PLAYOverlay(game_window, "Good job!  Now lets see how long you can go without blinking!", -1);
  waitForSpace();
#endif

  //-- 2. Read the video stream
  if( capture.isOpened() )
  {      
    while((char)cc != 'q')
    {
      //-- 3. Apply the classifier to the frame
      if( capture.read(frame) )
      {
            start = clock();

            flip(frame, frame, 1);
            num_faces = detectAndDisGAME_PLAY( frame );
//--(!) testing
//            imshow("Frame", &frame);

            if(num_faces > 0)
            {
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
 * @function detectAndDisGAME_PLAY
 */

//--(!) figure out how to make this faster!
// a: nuke current ubuntu 12.04 and install ubuntu 12.04LTS
// b: only pass mats by reference
int detectAndDisGAME_PLAY( Mat &frame )
{
    int numFaces = 0;

    cvtColor( frame, frame_gray, COLOR_BGR2GRAY );

    equalizeHist( frame_gray, frame_gray ); // maybe remove this for speed if unnecessary for quality
#ifdef DEBUG
    clock_t start;
    double dur;

    start=clock();
    face_cascade.detectMultiScale( frame_gray, faces, 1.1, 2, 0, Size(150, 150),Size(300,300) );
    dur = ( clock()-start )*1000./(double)CLOCKS_PER_SEC;
    cout << "Detection Durations\n";
    cout.precision(numeric_limits<double>::digits10);
    cout << "Face Cascade: " << fixed << dur << "ms\n";
#else
    //-- Detect faces
    face_cascade.detectMultiScale( frame_gray, faces, 1.1, 2, 0, Size(150, 150),Size(300,300) );
#endif

    for( size_t i = 0; i < faces.size(); i++ )
    {
        //-- Draw the face
 //       rectangle( frame, faces[i],  Scalar( 255, 0, 0 ), 2, 8, 0 );
        numFaces++;
        faceROI = frame_gray( faces[i] );
#ifdef DEBUG
        start = clock();
        eyes_cascade.detectMultiScale( faceROI, eyes, 1.1, 2, 0 |CV_HAAR_SCALE_IMAGE, Size(50,50),Size(80,80) );
        dur = ( clock()-start )*1000./(double)CLOCKS_PER_SEC;
        cout << "Eye Cascade: " << fixed << dur << "ms\t" << eyes.size() << "eyes found\n";
#else
        //-- In each face, detect eyes
        eyes_cascade.detectMultiScale( faceROI, eyes, 1.1, 2, 0 |CV_HAAR_SCALE_IMAGE, Size(50,50),Size(80,80) );
#endif
        if( eyes.size() == 2)
        {
         for( size_t j = 0; j < eyes.size(); j++ )
          { //-- Draw the eyes
            eye.x = eyes[j].x + faces[i].x;
            eye.y = eyes[j].y + faces[i].y;
            eye.width = eyes[j].width;
            eye.height = eyes[j].height;
//            rectangle( frame, eye, Scalar( 255, 0, 255 ), 2, 8, 0 );


#ifdef DEBUG
            start = clock();
            Point eye_center = findEyeCenter(faceROI, eyes[j], "Debug Window");
            dur = ( clock()-start )*1000./(double)CLOCKS_PER_SEC;
            cout << "Eye Center: " << fixed << dur << "ms\n";
#else
            // Find Eye Center
//            Point eye_center( eyes[j].width/2, eyes[j].height/2 );  // middle of eye rectangle
            Point eye_center = findEyeCenter(faceROI, eyes[j], "Debug Window");
#endif

            // take average of eye_center:eye width for both eyes
            if(j==0){
                x_frac = (double)eye_center.x / (double)eye.width;
                y_frac = (double)eye_center.y / (double)eye.height;
            } else {
                x_frac = 0.5 * (x_frac + ((double)eye_center.x / (double)eye.width));
                y_frac = 0.5 * (y_frac + ((double)eye_center.y / (double)eye.height));
            }

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
    cout <<"\n";

   //-- Show what you got
   imshow( window_name, frame );


#ifdef GAME_PLAY
   // TL: .44, .54
   // TR: .56, .54
   // BL: .44, .46
   // BR: .46, .46

   x_frac = 8.33*x_frac - 3.67;
   x_frac = min(max(x_frac, 0.0), 1.0);
   y_frac = 12.5*y_frac - 5.75;
   y_frac = min(max(y_frac, 0.0), 1.0);

   // Show where you're looking
   game_frame = Mat::zeros(height_screen, width_screen, CV_8UC3);
   gaze.x = x_frac * width_screen;
   gaze.y = y_frac * height_screen;
   circle(game_frame, gaze, 10, Scalar(255, 255, 255), 20);
   imshow(game_window, game_frame);
   printf("x = %f, y = %f\n", x_frac, y_frac);
#endif

   return numFaces;


}


Point getGaze(int x, int y) {
    Point gazeCal;
    game_frame = Mat::zeros(height_screen, width_screen, CV_8UC3);
    if(x >= 0) {
        gaze.x = x;
        gaze.y = y;
        circle(game_frame, gaze, 10, Scalar(255, 255, 255), 20);
    }
    imshow( game_window, game_frame );



    start = clock();
    do {
        waitKey(1);

        dur = (clock()-start) * 1000. / (double)CLOCKS_PER_SEC;

    } while(dur < 300);


    return gazeCal;
}

void waitForSpace() {
    while((char)cc != ' ') {
        cc = waitKey(1);
    }
    cc = 0;
}
