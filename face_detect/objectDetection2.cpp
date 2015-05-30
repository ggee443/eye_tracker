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

//#define GAME_PLAY
#define DEBUG

using namespace std;
using namespace cv;

/** Function Headers */
bool initializeGame();
int calibrate();
int detectAndDisplay( Mat &frame );
int detectFace( Rect &face );
Point detectEyeCenter();
Point getGaze(int x, int y);
void waitForSpace();
void initKalmanFilter(Point&);

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
Mat frame, frame_gray;
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
//double x_frac, y_frac;
double x_frac = 0.5;
double y_frac = 0.9;
Point gaze;

// TopLeft, TopRight, BottomLeft, BottomRight, Middle
Point gazeCalibrations[5];

int num_faces;

// Kalman objs
KalmanFilter RKF(4,2,0);
KalmanFilter LKF(4,2,0);



/**
 * @function main
 */
int main( void )
{
//  cap.open();  //sudo apt-get install v4l2ucp v4l-utils libv4l-dev
  // Timing Tests
  clock_t start;
  double dur;
  double avg_dur = 750.;
  cout.precision(numeric_limits<double>::digits10);


  //-- 1. Load the cascade
  if( !face_cascade.load( cascade_path + face_cascade_name ) ){ printf("--(!)objectDetection2:Error loading face_cascade files\n"); return -1; };
  if( !eyes_cascade.load( cascade_path + eyes_cascade_name ) ){ printf("--(!)objectDetection2:Error loading eyes_cascade files\n"); return -1; };


#ifdef DEBUG
  cout << "Game is initialized: " << boolalpha << initializeGame() << endl;

#endif

#ifdef GAME_PLAY
  // Game interface

  // Perform calibration
  displayOverlay(game_window, "Great! Next we are going to do some calibrations.  So get ready to stare at the circle for a bit.\n  Again, press 'space' to continue.", -1);
  waitForSpace();

  gazeCalibrations[0] = getGaze(20,60);
  gazeCalibrations[1] = getGaze(width_screen-20,60);
  gazeCalibrations[2] = getGaze(20,height_screen-20);
  gazeCalibrations[3] = getGaze(width_screen-20,height_screen-20);
  gazeCalibrations[4] = getGaze(width_screen/2,height_screen/2);



//  game_frame = Mat::zeros(height_screen, width_screen, CV_8UC3); // refresh?
//  imshow( game_window, game_frame ); waitKey(1);
  displayOverlay(game_window, "Good job!  Now lets see how long you can go without blinking!", -1);
  waitForSpace();
#endif

//  //-- 2. Read the video stream
//  if( cap.isOpened() )
//  {
//    while((char)cc != 'q')
//    {
//      //-- 3. Apply the classifier to the frame
//      if( cap.read(frame) )
//      {
//            start = clock();

//            flip(frame, frame, 1);
//            num_faces = detectAndDisplay( frame );
////--(!) testing
////            imshow("Frame", &frame);

//            if(num_faces > 0)
//            {
//                dur = (clock()-start) * 1000. / (double)CLOCKS_PER_SEC / (double) num_faces;
//                avg_dur = 0.9*avg_dur + 0.1*dur;
//                printf("Duration: %3.0fms\n", avg_dur);
//            }
//      }
//      else
//      { printf(" --(!) No captured frame -- Break!\n"); break; }

//      cc = waitKey(1);

//    }
//  }
  return 0;
}

// Reads videofeed, performs face detection, defines search space for eyes
bool initializeGame()
{
    Rect face_avg,face;
    clock_t ss;
    double dd, face_count;

    game_frame = Mat::zeros(height_screen, width_screen, CV_8UC3);
    imshow( game_window, game_frame ); waitKey(1);
    displayOverlay(game_window, "Welcome to the game!\n Press 'space' to continue.", -1);
    waitForSpace();

    ss = clock();
    do
    {
        if(cap.isOpened())
        {
            if(cap.read(frame))
            {
                //if(detectFace(face) <= 0){ continue; }
                detectFace(face);
                face_avg.width += face.width; face_avg.height += face.height;
                face_avg.x += face.x; face_avg.y += face.y;
                face_count++;
            }
            else { cerr << " --(!) Could not read frame\n" <<endl; return false; }
        }
        else{ cerr << " --(!) Could not open videostream\n" <<endl; return false; }

        dd = (clock()-ss) / (double)CLOCKS_PER_SEC;

    } while(dd < 1 && waitKey(1)!='q');

    faceROI.width = (int)( (double)face_avg.width/face_count );
    faceROI.height = (int)( (double)face_avg.height/face_count );
    faceROI.x = (int)( (double)face_avg.x/face_count );
    faceROI.y = (int)( (double)face_avg.y/face_count );


    eyeROI.width = faceROI.width; eyeROI.height = (int)(0.3*faceROI.height);    // only use upper half of face roi
    eyeROI.x = faceROI.x; eyeROI.y = (int)(1.37*faceROI.y);

#ifdef DEBUG
    cout << "initGame: averaging " << fixed << face_count << endl;

    printf("faceROI: %d,%d\n",faceROI.width,faceROI.height);
    printf("eyeROI: %d,%d\n",eyeROI.width,eyeROI.height);
    rectangle( frame, faceROI,  Scalar( 255, 0, 0 ), 1, 8, 0 );
    imshow("init complete", frame);waitKey(1000);
#endif

    if(calibrate() < 0){ return false; }

    return true;
}

// initializes gaze calibration points for screen corners and center
int calibrate()
{
    // Perform calibration
    displayOverlay(game_window, "Great! Next we are going to do some calibrations.  So get ready to stare at the circle for a bit.\n  Again, press 'space' to continue.", -1);
    waitForSpace();

    gazeCalibrations[0] = getGaze(20,60);
    gazeCalibrations[1] = getGaze(width_screen-20,60);
    gazeCalibrations[2] = getGaze(20,height_screen-20);
    gazeCalibrations[3] = getGaze(width_screen-20,height_screen-20);
    gazeCalibrations[4] = getGaze(width_screen/2,height_screen/2);

    return 0;
}



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

// takes a vector of two eyes
Point detectEyeCenter()
{
    Point eye_center;
    Rect eye;
    clock_t start;
    double dur;
    Mat eye_roi;

    for( size_t j = 0; j < eyes.size(); j++ )
    {
          //-- Draw the eyes
           eye.x = eyes[j].x + eyeROI.x;
           eye.y = eyes[j].y + eyeROI.y;
           eye.width = eyes[j].width;
           eye.height = eyes[j].height;
           rectangle( frame, eye, Scalar( 255, 0, 255 ), 1, 8, 0 );


#ifdef DEBUG
           start = clock();
           eye_roi = frame(eyeROI);
           eye_center = findEyeCenter(eye_roi, eyes[j], "Debug Window");
           dur = ( clock()-start )*1000./(double)CLOCKS_PER_SEC;
           cout << "Eye Center: " << fixed << dur << "ms\n";
#else
           // Find Eye Center
   //            Point eye_center( eyes[j].width/2, eyes[j].height/2 );  // middle of eye rectangle
           eye_center = findEyeCenter(eye_roi, eyes[j], "Debug Window");
#endif

           // Shift relative to eye
           eye_center.x += eyes[j].x + eyeROI.x;
           eye_center.y += eyes[j].y + eyeROI.y;

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
    cout <<"\n";
    //-- Show what you got
    imshow( window_name, frame );

    return eye_center;
}



Point getGaze(int x, int y) {
    Point gazeCal;
    clock_t start;
    double dur,temp;
    game_frame = Mat::zeros(height_screen, width_screen, CV_8UC3);
    if(x >= 0) {
        gaze.x = x;
        gaze.y = y;
        circle(game_frame, gaze, 10, Scalar(255, 255, 255), 20);
    }
    imshow( game_window, game_frame );waitKey(1);

    start = clock();
    do {
// detect eye centers and average them
        //read frame
        if(!cap.read(frame)) { cerr<< "unable to read frame\n"; return Point(); } //Point(NULL,NULL); }

        rectangle( frame, eyeROI,  Scalar( 0, 255, 0 ), 1, 8, 0 );
        imshow("eye gaze",frame);
#ifdef DEBUG
        temp = clock();
        eyes_cascade.detectMultiScale( frame(eyeROI), eyes, 1.1, 2, 0 |CV_HAAR_SCALE_IMAGE, Size(50,50),Size(80,80) );
        dur = ( clock()-temp )*1000./(double)CLOCKS_PER_SEC;
        cout << "Eye Cascade: " << fixed << dur << "ms\t" << eyes.size() << " eyes found\n";
#else
        //-- In each face, detect eyes
        eyes_cascade.detectMultiScale( frame(eyeROI), eyes, 1.1, 2, 0 |CV_HAAR_SCALE_IMAGE, Size(50,50),Size(80,80) );
#endif

        if( eyes.size() == 2)
        {
            gazeCal = detectEyeCenter();
        }

        dur = (clock()-start) / (double)CLOCKS_PER_SEC;

    } while(dur < 1 && waitKey(1)!='q');

#ifdef DEBUG
    cout << "gaze cal done\n";
#endif
    return gazeCal;
}

void waitForSpace() {
    while((char)cc != ' ') {
        cc = waitKey(1);
    }
    cc = 0;
}

void initKalmanFilter(Point &pp)
{



}

