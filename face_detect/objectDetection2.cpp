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
const double alpha_width=1., alpha_centers=1.;

using namespace std;
using namespace cv;

/** Function Headers */

void initKalmanFilter(Point&);
int startGame();

/** Global variables */
string window_name = "Capture - Face detection";
RNG rng(12345);

// Initialize Mem
Mat game_frame;
Mat playing_frame[2];
int cc;

// Search space
Rect gameROI;
Rect eyeROI;

//x range, y range
double gazeCalibrations[2];
double gaze_scale;

// Kalman objs for eye boxes:
// filter the average (x,y) of left and right eye box centers
KalmanFilter KF_CENTER(2,1,0);
//KalmanFilter KF_SIZE(1,1,0);
Mat state_center(2,1,CV_32F);//,state_size(1,1,CV_32F);
Mat measure_center(2,1,CV_32F);//,measure_size(1,1,CV_32F);


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
int main( int argc, char* argv[] )
{

//  cap.open();
  cout.precision(numeric_limits<double>::digits10);

  // Load game images
  playing_frame[0] = imread(staring_face,CV_LOAD_IMAGE_COLOR);
  playing_frame[1] = imread(lose_by_blink_face, CV_LOAD_IMAGE_COLOR);

  // Load the cascade
  cout << "loading eye cascades...\n";
  if( !eyes_cascade.load( eye_cascade_path + eyes_cascade_name ) ){ printf("--(!)objectDetection2:Error loading eyes_cascade files\n"); return -1; };
  cout << "initializing kalman...\n";
  // Initialize Kalman params - use average of left and right eye box centers as state param
//  double delta_x=1, delta_y=1;//, delta_size=0.5;
//  KF_CENTER.transitionMatrix = *(Mat_<float>(4,4,CV_32F) << 1,0,delta_x,0,   0,1,0,delta_y);
////  KF_SIZE.transitionMatrix = *(Mat_<float>(2,1,CV_32F) << 1,delta_size);


//  KF_CENTER.statePre.at<double>(0) = 0.5*( eyes[0].x+eyes[1].x );
//  KF_CENTER.statePre.at<double>(1) = 0.5*( eyes[0].y+eyes[1].y );

//  setIdentity(KF_CENTER.measurementMatrix);
//  setIdentity(KF_CENTER.processNoiseCov, Scalar::all(1e-4));
//  setIdentity(KF_CENTER.measurementNoiseCov, Scalar::all(10));
//  setIdentity(KF_CENTER.errorCovPost, Scalar::all(.1));

  if(argc>2)
  {
      cerr << "Invalid input: command takes only one arg, but more than one were passed.";
      return -1;
  }
  // user sets difficulty via gaze scale
  if(argc>1)
    gaze_scale = (double)*argv[1];
  else
    gaze_scale = 1.5;

  printf("Setting Gaze Scale to: %0.2f\n",gaze_scale);

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
    Point prev_eye_box_centers[2];
    Point cur_eye_box_centers[2];
    Point prev_eye_box_centers_avg, cur_eye_box_centers_avg;
    Rect temp_rect;

    // Kalman matrices
    Mat avg_center_prediction(2,1,CV_32F);
    Point_<float> predicted_point(0.f,0.f);

    // for display
    vector<Rect> bounds_to_disp(2,Rect(0,0,0,0));
    Point eye_centers_to_disp[2];
    Mat zoomed_eye_ss;

    double xBound = gaze_scale*gazeCalibrations[0];
    double yBound = gaze_scale*gazeCalibrations[1];
    // LPF params
    int prev_eye_box_width[2];
    int temp_int;


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

    displayOverlay(game_window, "Game Started", -1);
    imshow( game_window, game_frame ); waitKey(1);

    cout << "searching for eyes...\n";
    while((char)cc != 'q'){
        // Read Frame / Pre-Process
        if(!capture.read(frame)) { cerr<< "unable to read frame\n"; return -1; }
        // NOTE: eye center estimation REQUIRES grey scale
        eye_search_space = frame(eyeROI);
        flip(eye_search_space, eye_search_space, 1);
#ifdef CHECK_GAZE
        cvtColor( eye_search_space, eye_search_space, COLOR_BGR2GRAY );
#endif
        // Find Eyes
        eyes_cascade.detectMultiScale(eye_search_space, eyes, 1.1, 2, 0 |CV_HAAR_SCALE_IMAGE, Size(50,50),Size(80,80));

      // Catch a blink
       if(eyes.size() == 0) {
           sprintf(overlay_str, "You Blinked!\nScore: %d\nPlay Again? (y/n)", your_score);
           break;
       }


       if( eyes.size() == 2) {

    #ifdef CHECK_GAZE
       // Determine which eye is which, discard if both on one side
           int mean0 = eyes[0].x + eyes[0].width/2;
           int mean1 = eyes[1].x + eyes[1].width/2;
           // swap eyes s.t. left eye is always in index 0
           if(mean0 < eyeROI.width/2 && mean1 > eyeROI.width/2) {
            //do nothing
           }
           else if(mean0 > eyeROI.width/2 && mean1 < eyeROI.width/2) {
               temp_rect = eyes[0];
               eyes[0] = eyes[1];
               eyes[1] = temp_rect;
           } else {
               continue;
           }

           // FILTER EYE WIDTHS AND CENTTERS
           //
           //
           if(your_score == 0){
               printf("Initialize\n\n");
               // Initialize Widths
               prev_eye_box_width[0] = eyes[0].width;
               prev_eye_box_width[1] = eyes[1].width;

               // Initialize Centers
               for(size_t j=0; j<2; j++){
                   prev_eye_box_centers[j].x = eyes[j].x + eyes[j].width/2;
                   prev_eye_box_centers[j].y = eyes[j].y + eyes[j].height/2;
               }
               // Average
               prev_eye_box_centers_avg.x = (prev_eye_box_centers[0].x + prev_eye_box_centers[1].x) / 2;
               prev_eye_box_centers_avg.y = (prev_eye_box_centers[0].y + prev_eye_box_centers[1].y) / 2;
           }
           else {
               for(size_t j=0; j<2; j++){
                   printf("width before:%d\n", eyes[j].width);
                   // Filter Width & Height
                   temp_int = (int)(alpha_width*eyes[j].width) +
                              (int)((1.-alpha_width)*prev_eye_box_width[j]); // lpf width
                   prev_eye_box_width[j] = eyes[j].width; //update prev
                   // Update Eye Width/Height
                   eyes[j].width = temp_int;
                   eyes[j].height = temp_int;
                   printf("width after:%d\n", eyes[j].width);

                   // Find Current Centers
                   cur_eye_box_centers[j].x = eyes[j].x + eyes[j].width/2;
                   cur_eye_box_centers[j].y = eyes[j].y + eyes[j].height/2;
               }

               // Find Current Center
               cur_eye_box_centers_avg.x =  (cur_eye_box_centers[0].x + cur_eye_box_centers[1].x) / 2;
               cur_eye_box_centers_avg.y = (cur_eye_box_centers[0].y + cur_eye_box_centers[1].y) / 2;

               // Filter Centers
               printf("Centers\n\n");
               for(size_t j=0; j<2; j++){
                   // X Dim
                   temp_int = cur_eye_box_centers[j].x; // update using unfiltered center --> FIR
                   cur_eye_box_centers[j].x *= alpha_centers;
                   cur_eye_box_centers[j].x += (int)((1.-alpha_centers) * (prev_eye_box_centers[j].x +
                                            cur_eye_box_centers_avg.x - prev_eye_box_centers_avg.x));
                   // Update Prev
                   prev_eye_box_centers[j].x = temp_int;
                   printf("Update Eye X\n\n");
                   // Update Eye X (TOPLEFT)
                   printf("x: %d\n", eyes[j].x);
                   printf("y: %d\n", eyes[j].y);
                   printf("w: %d\n", eyes[j].width);
                   printf("h: %d\n", eyes[j].height);
                   printf("Val = %d\n", cur_eye_box_centers[j].x - eyes[j].width/2);
                   eyes[j].x = cur_eye_box_centers[j].x - eyes[j].width/2;

                   // Y Dim
                   temp_int = cur_eye_box_centers[j].y;
                   cur_eye_box_centers[j].y *= alpha_centers;
                   cur_eye_box_centers[j].y += (int)((1.-alpha_centers) * (prev_eye_box_centers[j].y +
                                            cur_eye_box_centers_avg.y - prev_eye_box_centers_avg.y));
                   // Update Prev
                   prev_eye_box_centers[j].y = temp_int;
                   printf("Update Eye Y\n\n");
                   // Update Eye Y (TOP-LEFT)
                   printf("x: %d\n", eyes[j].x);
                   printf("y: %d\n", eyes[j].y);
                   printf("w: %d\n", eyes[j].width);
                   printf("h: %d\n", eyes[j].height);
                   printf("Val = %d\n", cur_eye_box_centers[j].y - eyes[j].height/2);
                   eyes[j].y = cur_eye_box_centers[j].y - eyes[j].height/2;
               }

               // Update Prev Center Avg
               prev_eye_box_centers_avg.x = cur_eye_box_centers_avg.x;
               prev_eye_box_centers_avg.y = cur_eye_box_centers_avg.y;

           }
//           prev_eye_box_centers[0] = eyes[0],prev_eye_box_centers[1] = eyes[1];



//       // predict eyes center average
//       avg_center_prediction = KF_CENTER.predict();
//       predicted_point.x = avg_center_prediction.at<float>(0);
//       predicted_point.y = avg_center_prediction.at<float>(1);
//       printf("Kalman prediction: (%.2f,%.2f)\n",predicted_point.x,predicted_point.y);


//       // update measurement and prediction
//       measure_center(0) = 0.5*(eyes[0].x+eyes[1].x);
//       measure_center(1) = 0.5*(eyes[0].y+eyes[1].y);
//       KF_CENTER.correct(measure_center);
//       printf("measured eye center avg: (%.2f,%.2f)\n",0.5*(eyes[0].x+eyes[1].x),0.5*(eyes[0].y+eyes[1].y));

    printf("About to check gaze\n");

//    #ifdef CHECK_GAZE
       // Check Gaze

           // Find Eye Center
           eye_centers[0] = findEyeCenter(eye_search_space, eyes[0], "_");
           eye_centers[1] = findEyeCenter(eye_search_space, eyes[1], "_");

           printf("Finished FIC\n");


           #ifdef DEBUG
               // shift eye_centers to midpoint of eye boxes
               eye_centers[0].x += eyes[0].x;
               eye_centers[0].y += eyes[0].y;
               eye_centers[1].x += eyes[1].x;
               eye_centers[1].y += eyes[1].y;

//               eye_centers_to_disp[0].x = eye_centers[0].x + eye_search_space.rows;
//               eye_centers_to_disp[0].x = eye_centers[0].x + eye_search_space.rows;

               // Left eye region and center
               rectangle( eye_search_space, eyes[0], Scalar( 255, 0, 255 ), 1, 8, 0 );
//               rectangle( eye_search_space, eyes[0], Scalar( 255, 0, 255 ), 1, 8, 0 );
               circle(eye_search_space, eye_centers[0], 2, Scalar(255,255,255), 1);
               // Right eye region and center
               rectangle( eye_search_space, eyes[1], Scalar( 255, 0, 255 ), 1, 8, 0 );
               circle(eye_search_space, eye_centers[1], 2, Scalar(255,255,255), 1);

               // display eye centers on game screen
               circle(game_frame, eye_centers[0], 2, Scalar(255,255,255), 1);

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

               // resize eye_search_space and display
               resize(eye_search_space,zoomed_eye_ss,Size(),3.f,3.f,CV_INTER_LINEAR);
               imshow( "Zoomed Eye Search Space",zoomed_eye_ss);
               cc=waitKey(1);

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

    #endif
        }
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



// Kalman filtering

// filter eye box center and size
//   convert x,y to center point
//   filter width only and duplicate
