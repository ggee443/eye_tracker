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

#include <string>

#include "findEyeCenter.h"
#include "constants.h"
#include "helpers.h"
#include "globals.h"
#include "initialize.h"

#define CHECK_GAZE
#define DEBUG

using namespace std;
using namespace cv;

/** Function Headers */

void initKalmanFilter(Point&);
int startGame();

/** Global variables */
string window_name = "Capture - Face detection";
RNG rng(12345);


//cv::CascadeClassifier eyes_cascade;

// Initialize Mem
Mat game_frame;
Mat playing_frame[2];
int cc;

// Search space
Rect gameROI;
Rect eyeROI;

//x range, y range
double gazeCalibrations[2];

// Kalman objs
KalmanFilter RKF(4,2,0);
KalmanFilter LKF(4,2,0);

bool mid_not_corners = true;

char overlay_str[100];
int score;

// from initialize.h
extern CascadeClassifier eyes_cascade;
extern VideoCapture capture; // watch this
extern Mat frame, eye_search_space;
extern vector<Rect> eyes;

/**
 * @function main
 */
int main( void )
{
//  cap.open();
  cout.precision(numeric_limits<double>::digits10);

  // Load game images
  playing_frame[0] = imread(staring_face,CV_LOAD_IMAGE_COLOR);
  playing_frame[1] = imread(lose_by_blink_face, CV_LOAD_IMAGE_COLOR);

  // Load the cascade
  if( !eyes_cascade.load( eye_cascade_path + eyes_cascade_name ) ){ printf("--(!)objectDetection2:Error loading eyes_cascade files\n"); return -1; };

  // Initialize Game
  cout << "Game is initialized: " << boolalpha << initializeGame(game_frame, eyeROI, gazeCalibrations) << endl;


  // Play Game
  do {
    startGame();

    cc = 0;
    while((char)cc != 'y' && (char)cc != 'n') {
        cc = waitKey(1);
    }


  } while((char)cc != 'n');


  return 0;
}

void initKalmanFilter(Point &pp) {
    return;
}



int startGame() {
    printf("Game has started\n\n");

    int your_score = 0;
    Point eye_centers[2];
    vector<Rect> bounds_to_disp(2,Rect(0,0,0,0));

    double xBound = 1.5*gazeCalibrations[0];
    double yBound = 1.5*gazeCalibrations[1];

    // Prompt to start
    game_frame = Mat::zeros(height_screen, width_screen, CV_8UC3);
    displayOverlay(game_window, "Great!  Let the game begin! Press 'space' to start", -1);

    cc = 0;
    while((char)cc != ' ') {
        if(!capture.read(frame)) { cerr<< "unable to read frame\n"; return -1; }
        eye_search_space = frame(eyeROI);
        flip(eye_search_space, eye_search_space, 1);

        eye_search_space.copyTo(game_frame(Rect(width_screen/2 - eye_search_space.cols/2,
                                     0.35*height_screen - eye_search_space.rows/2,
                                     eye_search_space.cols, eye_search_space.rows)));
        imshow( game_window, game_frame );
        cc = waitKey(1);
    }
    cc = 0;

    playing_frame[0].copyTo(game_frame(Rect(width_screen/2 - playing_frame[0].cols/2,
                                         0.35*height_screen - playing_frame[0].rows/2,
                                         playing_frame[0].cols, playing_frame[0].rows)));
    imshow( game_window, game_frame );
    cc = waitKey(1);

    cout << "searching for eyes...\n";
    while((char)cc != 'q'){
        // Read Frame / Pre-Process
        if(!capture.read(frame)) { cerr<< "unable to read frame\n"; return -1; }
        // NOTE: eye center estimation REQUIRES grey scale
        eye_search_space = frame(eyeROI);
        flip(eye_search_space, eye_search_space, 1);
        cvtColor( eye_search_space, eye_search_space, COLOR_BGR2GRAY );

        // Find Eyes
        eyes_cascade.detectMultiScale(eye_search_space, eyes, 1.1, 2, 0 |CV_HAAR_SCALE_IMAGE, Size(50,50),Size(80,80));

      // Catch a blink
       if(eyes.size() == 0) {
           sprintf(overlay_str, "You Blinked!\nScore: %d\nPlay Again? (y/n)", your_score);
           break;
       }

    #ifdef CHECK_GAZE
       // Check Gaze
       if( eyes.size() == 2) {
           // Find Eye Center
           eye_centers[0] = findEyeCenter(eye_search_space, eyes[0], "_");
           eye_centers[1] = findEyeCenter(eye_search_space, eyes[1], "_");

           #ifdef DEBUG
               // shift eye_centers to midpoint of eye boxes
               eye_centers[0].x += eyes[0].x;
               eye_centers[0].y += eyes[0].y;
               eye_centers[1].x += eyes[1].x;
               eye_centers[1].y += eyes[1].y;

               // Left eye region and center
               rectangle( eye_search_space, eyes[0], Scalar( 255, 0, 255 ), 1, 8, 0 );
               rectangle( eye_search_space, eyes[0], Scalar( 255, 0, 255 ), 1, 8, 0 );
               circle(eye_search_space, eye_centers[0], 2, Scalar(255,255,255), 1);
               // Right eye region and center
               rectangle( eye_search_space, eyes[1], Scalar( 255, 0, 255 ), 1, 8, 0 );
               circle(eye_search_space, eye_centers[1], 2, Scalar(255,255,255), 1);

               // Set Boundaries to Display
               bounds_to_disp[0].width = 2.*xBound*eyes[0].width;
               bounds_to_disp[0].height = 2.*yBound*eyes[0].height;
               bounds_to_disp[0].x = eyes[0].x+(0.5-xBound)*eyes[0].width;
               bounds_to_disp[0].y = eyes[0].y+(0.5-yBound)*eyes[0].height;
               bounds_to_disp[1].width = 2*xBound*eyes[1].width;
               bounds_to_disp[1].height = 2*yBound*eyes[1].height;
               bounds_to_disp[1].x = eyes[1].x+(0.5-xBound)*eyes[1].width;
               bounds_to_disp[1].y = eyes[1].y+(0.5-yBound)*eyes[1].height;

               // Draw boundary region
               rectangle( eye_search_space, bounds_to_disp[0], Scalar(255,255,255), 1, 8, 0 );
               rectangle( eye_search_space, bounds_to_disp[1], Scalar(255,255,255), 1, 8, 0 );
               imshow( "Eye Search Space", eye_search_space ); cc=waitKey(1);

               // undo shift eye_centers
               eye_centers[0].x -= eyes[0].x;
               eye_centers[0].y -= eyes[0].y;
               eye_centers[1].x -= eyes[1].x;
               eye_centers[1].y -= eyes[1].y;
           #endif

           // Check in bounds
           if(eye_centers[0].x < (0.5-xBound)*eyes[0].width) {
               sprintf(overlay_str, "You Looked Left!\nScore: %d\nPlay Again? (y/n)", your_score);
               printf("eye_width: %d\n", eyes[0].width);
               printf("xBoundL: %d\n", (int)((0.5-xBound)*eyes[0].width));
               printf("xEyeCenter: %d\n", eye_centers[0].x);

               break;
           }
           if(eye_centers[0].x > (0.5+xBound)*eyes[0].width) {
               sprintf(overlay_str, "You Looked Right!\nScore: %d\nPlay Again? (y/n)", your_score);
               printf("eye_width: %d\n", eyes[0].width);
               printf("xBoundR: %d\n", (int)((0.5+xBound)*eyes[0].width));
               printf("xEyeCenter: %d\n", eye_centers[0].x);

               break;
           }
           if(eye_centers[0].y < (0.5-yBound)*eyes[0].height) {
               sprintf(overlay_str, "You Looked Up!\nScore: %d\nPlay Again? (y/n)", your_score);
               printf("eye_width: %d\n", eyes[0].height);
               printf("yBoundU: %d\n", (int)((0.5-yBound)*eyes[0].height));
               printf("yEyeCenter: %d\n", eye_centers[0].y);
               break;
           }
           if(eye_centers[0].y > (0.5+yBound)*eyes[0].height) {
               sprintf(overlay_str, "You Looked Down!\nScore: %d\nPlay Again? (y/n)", your_score);
               printf("eye_width: %d\n", eyes[0].height);
               printf("yBoundD: %d\n", (int)((0.5+yBound)*eyes[0].height));
               printf("yEyeCenter: %d\n", eye_centers[0].y);
               break;
           }


           // Check in bounds
           if(eye_centers[1].x < (0.5-xBound)*eyes[1].width) {
               sprintf(overlay_str, "You Looked Left!\nScore: %d\nPlay Again? (y/n)", your_score);
               break;
           }
           if(eye_centers[1].x > (0.5+xBound)*eyes[1].width) {
               sprintf(overlay_str, "You Looked Right!\nScore: %d\nPlay Again? (y/n)", your_score);
               break;
           }
           if(eye_centers[1].y < (0.5-yBound)*eyes[1].height) {
               sprintf(overlay_str, "You Looked Up!\nScore: %d\nPlay Again? (y/n)", your_score);
               break;
           }
           if(eye_centers[1].y > (0.5+yBound)*eyes[1].height) {
               sprintf(overlay_str, "You Looked Down!\nScore: %d\nPlay Again? (y/n)", your_score);
               break;
           }

        }
    #endif

        your_score++;

        cc = waitKey(1);
    }



    // clear game screen
    game_frame = Mat::zeros(height_screen, width_screen, CV_8UC3);
    // load and display losing face :(

    playing_frame[1].copyTo(game_frame(Rect(width_screen/2 - playing_frame[1].cols/2,
                                         0.35*height_screen - playing_frame[1].rows/2,
                                         playing_frame[1].cols, playing_frame[1].rows)));
    displayOverlay(game_window, overlay_str, -1);
    imshow( game_window, game_frame ); waitKey(1);




    return your_score;
}
