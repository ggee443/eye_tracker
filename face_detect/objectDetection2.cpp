/**
 * @file objectDetection2.cpp
 * @author A. Huaman ( based in the classic facedetect.cpp in samples/c )
 * @brief A simplified version of facedetect.cpp, show how to load a cascade classifier and how to find objects (Face + eyes) in a video stream - Using LBP here
 */
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>

#include <iostream>
#include <stdio.h>
#include <sstream>

#include "findEyeCenter.h"
#include "findEyeCorner.h"

#define DEBUG

using namespace std;
using namespace cv;

/** Function Headers */
bool initializeGame();
bool findFaceROI(int averaging_duration);
int calibrate();
int detectAndDisplay( Mat &frame );
int detectFace( Rect &face );
void getGaze(int x, int y, int index);
void waitForSpace();
void initKalmanFilter(Point&);

void drawEyeCenter(Point eye_center);

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
vector<Rect> faces, eyes;
Mat frame, frame_gray, eye_search_space;
Point crosshair1, crosshair2;
int cc;

VideoCapture cap(0);

// Search space
Rect gameROI;
Rect faceROI, eyeROI;

// Game
int width_screen = 1590;
int height_screen = 1100;
Mat game_frame;
Point gaze;

// TopLeft, TopRight, BottomLeft, BottomRight, Middle
double* gazeCalibrations[5];

// Kalman objs
KalmanFilter RKF(4,2,0);
KalmanFilter LKF(4,2,0);



/**
 * @function main
 */
int main( void )
{
//  cap.open();
  cout.precision(numeric_limits<double>::digits10);

  //-- 1. Load the cascade
  if( !face_cascade.load( cascade_path + face_cascade_name ) ){ printf("--(!)objectDetection2:Error loading face_cascade files\n"); return -1; };
  if( !eyes_cascade.load( cascade_path + eyes_cascade_name ) ){ printf("--(!)objectDetection2:Error loading eyes_cascade files\n"); return -1; };


#ifdef DEBUG
  for(int j = 0; j<5; j++) {
      double new_gaze_avg[2];
      gazeCalibrations[j] = new_gaze_avg;
  }

  cout << "Game is initialized: " << boolalpha << initializeGame() << endl;
#endif

  return 0;
}




// Reads videofeed, performs face detection, defines search space for eyes
bool initializeGame()
{
    // Starting Screen
    game_frame = Mat::zeros(height_screen, width_screen, CV_8UC3);
    imshow( game_window, game_frame ); waitKey(1);
    displayOverlay(game_window, "Welcome to the Staring Game!\n\n Press 'space' to continue.", -1);
    waitForSpace();

    // Pre - Face Finding
    displayOverlay(game_window, "Okay we are going to try to find your pretty face,\n so get comfortable because you can't move after this!\n\nPress 'space' to continue.", -1);
    waitForSpace();

    // Find Face
    displayOverlay(game_window, "Working...", -1); imshow( game_window, game_frame ); waitKey(1);
    if(findFaceROI(1) < 0){ return false; }
    rectangle( frame, faceROI,  Scalar( 255, 0, 0 ), 1, 8, 0 );
    frame.copyTo(game_frame(Rect(width_screen/2 - frame.cols/2, height_screen/2 - frame.rows/2, frame.cols, frame.rows)));
    displayOverlay(game_window, "How does this look?\n(The location of the box, we know you're beautiful)\n\nPress 'space' to continue, 'r' to try again.", -1);
    imshow( game_window, game_frame ); waitKey(1);
    while((char)cc != ' ' && (char)cc != 'r') {
        cc = waitKey(1);
    }

    // Find Face Again Until Good
    while((char)cc == 'r') {
        cc = 0;
        displayOverlay(game_window, "Working...", -1); imshow( game_window, game_frame ); waitKey(1);
        if(findFaceROI(1) < 0){ return false; }
        rectangle( frame, faceROI,  Scalar( 255, 0, 0 ), 1, 8, 0 );
        frame.copyTo(game_frame(Rect(width_screen/2 - frame.cols/2, height_screen/2 - frame.rows/2, frame.cols, frame.rows)));
        displayOverlay(game_window, "Is this better?\n\nPress 'space' to continue, 'r' to try again.", -1);
        imshow( game_window, game_frame ); waitKey(1);
        while((char)cc != ' ' && (char)cc != 'r') {
            cc = waitKey(1);
        }
    }
    cc = 0;

    // Set eyeROI as restricted portion of faceROI
    eyeROI.width = faceROI.width; eyeROI.height = (int)(0.6*faceROI.height);    // only use upper half of face roi
    eyeROI.x = faceROI.x; eyeROI.y = (int)(1.2*faceROI.y);

    // Calibrate Gaze
    if(calibrate() < 0){ return false; }

    return true;
}

void waitForSpace() {
    while((char)cc != ' ') {
        cc = waitKey(1);
    }
    cc = 0;
}

// Read frames for "averaging_duration" seconds
// and update faceROI such that it boxes the face
bool findFaceROI(int averaging_duration){
    Rect face_avg,face;
    clock_t ss;
    double dd;
    double face_count = 0.;

    ss = clock();
    do
    {
        if(cap.isOpened())
        {
            if(cap.read(frame))
            {
                flip(frame, frame, 1);

                // Find Face
                detectFace(face);

                // Add detected Rect to average
                face_avg.width += face.width; face_avg.height += face.height;
                face_avg.x += face.x; face_avg.y += face.y;
                face_count++;
            }
            else { cerr << " --(!) Could not read frame\n" <<endl; return false; }
        }
        else{ cerr << " --(!) Could not open videostream\n" <<endl; return false; }

        dd = (clock()-ss) / (double)CLOCKS_PER_SEC;

    } while((dd < averaging_duration || face_count==0) && waitKey(1)!='q');

    faceROI.width = (int)( (double)face_avg.width/face_count );
    faceROI.height = (int)( (double)face_avg.height/face_count );
    faceROI.x = (int)( (double)face_avg.x/face_count );
    faceROI.y = (int)( (double)face_avg.y/face_count );

#ifdef DEBUG
    cout << "initGame: averaging " << fixed << face_count << endl;

    printf("faceROI: %d,%d\n",faceROI.width,faceROI.height);
    printf("eyeROI: %d,%d\n",eyeROI.width,eyeROI.height);
#endif

    return true;
}

// Used in findFaceROI()
int detectFace(Rect &face)
{
    cvtColor( frame, frame_gray, COLOR_BGR2GRAY );

    equalizeHist( frame_gray, frame_gray ); // maybe remove this for speed if unnecessary for quality
#ifdef DEBUG
    clock_t start;
    double dur;

    start=clock();
    face_cascade.detectMultiScale( frame_gray, faces, 1.1, 2, 0, Size(150, 150),Size(300,300) );
    dur = ( clock()-start )*1000./(double)CLOCKS_PER_SEC;
    cout << "Detection Durations\n";
    cout << "Face Cascade: " << fixed << dur << "ms\n";
#else
    //-- Detect faces
    face_cascade.detectMultiScale( frame_gray, faces, 1.1, 2, 0, Size(150, 150),Size(300,300) );
#endif

    // for now, assume only one face is in frame
    // later, if mult faces are in frame, then take face closest to center of frame

    if(faces.size()==0){ face = Rect(); return -1; } //no face was found
    face = faces[0]; return 0;
}

// initializes gaze calibration points for screen corners and center
int calibrate()
{
    // Pre - Gaze Calibration
    displayOverlay(game_window, "Great!  Now we are going to do some gaze calibrations.\nYou need to stare at the white dot.\n\nPress 'space' to continue.", -1); imshow( game_window, game_frame ); waitKey(1);
    waitForSpace();

    // Perform calibration
    getGaze(20,50,0);
    getGaze(width_screen-20,50,1);
    getGaze(20,height_screen-20,2);
    getGaze(width_screen-20,height_screen-20,3);
//    gazeCalibrations[4] = getGaze(width_screen/2,height_screen/2);

    return 0;
}


void getGaze(int x, int y, int index) {
    double* gaze_avg = gazeCalibrations[index];
    double gaze_count = 0;
    Point eye_center;
    clock_t start;
    double dur;

    // Display the circle
    game_frame = Mat::zeros(height_screen, width_screen, CV_8UC3);
    if(x >= 0) {
        gaze.x = x;
        gaze.y = y;
        circle(game_frame, gaze, 10, Scalar(255, 255, 255), 20);
    }
    displayOverlay(game_window, "", -1); imshow( game_window, game_frame ); waitKey(1);

    // Initialize gaze value
    gaze_avg[0] = 0.;
    gaze_avg[1] = 0.;

    // Start timer
    start = clock();

    // detect eye centers and average them
    do {
        // Read frame
        if(!cap.read(frame)) { cerr<< "unable to read frame\n"; return; }
        flip(frame, frame, 1);
        cvtColor( frame, frame, COLOR_BGR2GRAY );
        eye_search_space = frame(eyeROI);

        // Find Eyes
        eyes_cascade.detectMultiScale(eye_search_space, eyes, 1.1, 2, 0 |CV_HAAR_SCALE_IMAGE, Size(50,50),Size(80,80));

        if( eyes.size() == 2) {
            // For each eye
            for(size_t j = 0; j < eyes.size(); j++) {

                // Find Eye Center
                eye_center = findEyeCenter(eye_search_space, eyes[j], "Debug Window");

                // Update gaze_avg
                gaze_avg[0] += (double)eye_center.x / (double)eyes[j].width;
                gaze_avg[1] += (double)eye_center.y / (double)eyes[j].height;
                gaze_count++;

//                // Shift relative to eye
//                eye_center.x += eyes[j].x + eyeROI.x;
//                eye_center.y += eyes[j].y + eyeROI.y;
            }
        }



/*
#ifdef DEBUG
        rectangle( frame, faceROI,  Scalar( 255, 0, 0 ), 1, 8, 0 );
        rectangle( frame, eyeROI,  Scalar( 0, 255, 0 ), 1, 8, 0 );

        Rect eye;
        for( size_t j = 0; j < eyes.size(); j++ )
        {
              //-- Draw the eyes
               eye.x = eyes[j].x + eyeROI.x;
               eye.y = eyes[j].y + eyeROI.y;
               eye.width = eyes[j].width;
               eye.height = eyes[j].height;
               rectangle( frame, eye, Scalar( 255, 0, 255 ), 1, 8, 0 );

               // Draw Eye Center
               // (only one eye! put in "for each eye" loop to see both)
               drawEyeCenter(eye_center);
        }



        imshow("eye gaze",frame);
        cout << "Eye Cascade: " << fixed << dur << "ms\t" << eyes.size() << " eyes found\n";
#endif
*/

        // Update timer
        dur = (clock()-start) / (double)CLOCKS_PER_SEC;

    } while(dur < 10 && waitKey(1)!='q');


    gaze_avg[0] /= gaze_count;
    gaze_avg[1] /= gaze_count;

    cout << "Gaze Average: (" << gaze_avg[0] << ", " << gaze_avg[1] << ") over " << (int)gaze_count << "frames\n";

    return;
}







// Only for debugging
void drawEyeCenter(Point eye_center){
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







void initKalmanFilter(Point &pp)
{

}

