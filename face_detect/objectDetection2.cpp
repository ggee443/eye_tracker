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
#include "findEyeCorner.h"
#include "constants.h"
#include "helpers.h"
#include "globals.h"
#include "initialize.h"

#define DEBUG

using namespace std;
using namespace cv;

/** Function Headers */
int calibrate();
void getGaze(int x, int y, int index);
void initKalmanFilter(Point&);
int startGame();
void sleep_for(unsigned int seconds);
void checkCalibrations();
void checkCalibrationsMid();
void testFindEyeCenter();

void drawEyeCenter(Point eye_center);

/** Global variables */
string window_name = "Capture - Face detection";
cv::CascadeClassifier eyes_cascade;

RNG rng(12345);

// Initialize Mem
vector<Rect> eyes;
Mat frame, eye_search_space;
Point crosshair1, crosshair2;
int cc;

// Search space
Rect gameROI;
Rect eyeROI;
Rect leftEye, rightEye;

// Game
Point gaze;

// x range, y range
double gazeCalibrations[2];

// Kalman objs
KalmanFilter RKF(4,2,0);
KalmanFilter LKF(4,2,0);

bool mid_not_corners = true;

char overlay_str[100];
int score;




/**
 * @function main
 */
int main( void )
{
   Mat game_frame;


//  cap.open();
  cout.precision(numeric_limits<double>::digits10);

  // Load the cascade
  if( !eyes_cascade.load( eye_cascade_path + eyes_cascade_name ) ){ printf("--(!)objectDetection2:Error loading eyes_cascade files\n"); return -1; };


  // Initialize Game
  cout << "Game is initialized: " << boolalpha << initializeGame(game_frame, eyeROI) << endl;


//  // Play Game
//  do {
//    startGame();

//    cc = 0;
//    while((char)cc != 'y' && (char)cc != 'n') {
//        cc = waitKey(1);
//    }


//  } while((char)cc != 'n');


  return 0;
}


/*


// initializes gaze calibration points for screen corners and center
int calibrate() {
    if(mid_not_corners) {
        // Middle
        getGaze(width_screen/2,height_screen/2,4);
    } else {
        // Corners
        getGaze(20,50,0);
        getGaze(width_screen-20,50,1);
        getGaze(20,height_screen-20,2);
        getGaze(width_screen-20,height_screen-20,3);
    }

    return 0;
}

// Update gazeCalibrations to have fractional (x,y) coordinates
// corresponding to staring at location x,y on the screen
void getGaze(int x, int y, int index) {
    double gaze_count = 0.;
    Point eye_center_L, eye_center_R;
    clock_t start;
    double dur, temp,  fraction[4]; //fr[0:2) --> L(x/W,y/H); fr[2:4) --> R(x/W,y/H)


    // Display the circle
    game_frame = Mat::zeros(height_screen, width_screen, CV_8UC3);
    if(x >= 0) {
        gaze.x = x;
        gaze.y = y;
        circle(game_frame, gaze, 10, Scalar(0, 0, 255), 20);
    }
    displayOverlay(game_window, "", -1); imshow( game_window, game_frame ); waitKey(1);
    sleep_for(50);

    circle(game_frame, gaze, 10, Scalar(0, 255, 255), 20);
    imshow( game_window, game_frame ); waitKey(1);

    // Initialize gaze value
    gazeCalibrationsLeftEye[index][0] = 0.;
    gazeCalibrationsLeftEye[index][1] = 0.;
    gazeCalibrationsRightEye[index][0] = 0.;
    gazeCalibrationsRightEye[index][1] = 0.;

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
            // Determine which eye is which, discard if both on one side
            int mean0 = eyes[0].x + eyes[0].width/2;
            int mean1 = eyes[1].x + eyes[1].width/2;
            if(mean0 < eyeROI.width/2 && mean1 > eyeROI.width/2) {
                leftEye = eyes[0];
                rightEye = eyes[1];
            }
            else if(mean0 > eyeROI.width/2 && mean1 < eyeROI.width/2) {
                leftEye = eyes[1];
                rightEye = eyes[0];
            } else {
                continue;
            }

            // Find Left Eye Center
            eye_center_L = findEyeCenter(eye_search_space, leftEye, "_");
            // Find Right Eye Center
            eye_center_R = findEyeCenter(eye_search_space, rightEye, "_");

            printf("calibration - l_eye_center: (%d,%d); eye_box_dims: %dx%d\n", \
                   eye_center_L.x,\
                   eye_center_L.y,\
                   leftEye.width,\
                   leftEye.height);


            // Find Eye Center Fractions
            fraction[0] = (double)eye_center_L.x / (double)leftEye.width;
            fraction[1] = (double)eye_center_L.y / (double)leftEye.height;
            fraction[2] = (double)eye_center_R.x / (double)rightEye.width;
            fraction[3] = (double)eye_center_R.y / (double)rightEye.height;

            // Check center is not excessively close to ROI border (eyebrows or something)
            double border = 0.25;
            if(fraction[0] < border || fraction[0] > (1.-border) ||
                    fraction[1] < border || fraction[1] > (1.-border) ||
                    fraction[2] < border || fraction[2] > (1.-border) ||
                    fraction[3] < border || fraction[3] > (1.-border)) {
                continue;
            }

            if(!mid_not_corners) {
                // Find Average Value of Point

                // Accumulate eye_center ratios for each eye
                // Update gazeCalibrationsLeftEye
                gazeCalibrationsLeftEye[index][0] += fraction[0];
                gazeCalibrationsLeftEye[index][1] += fraction[1];
                // Update gazeCalibrationsRightEye
                gazeCalibrationsRightEye[index][0] += fraction[2];
                gazeCalibrationsRightEye[index][1] += fraction[3];
            } else {
                // Find max distance from center
                gazeCalibrationsLeftEye[index][0] = max( abs(0.5-fraction[0]), gazeCalibrationsLeftEye[index][0] );
                gazeCalibrationsLeftEye[index][1] = max( abs(0.5-fraction[1]), gazeCalibrationsLeftEye[index][1] );

                gazeCalibrationsRightEye[index][0] = max( abs(0.5-fraction[2]), gazeCalibrationsRightEye[index][0] );
                gazeCalibrationsRightEye[index][1] = max( abs(0.5-fraction[3]), gazeCalibrationsRightEye[index][1] );
            }

            gaze_count++;

            // Report Immediate Center Fractions
            printf("L: (%1.2f,%1.2f)\n", fraction[0], fraction[1]);
            printf("R: (%1.2f,%1.2f)\n", fraction[2], fraction[3]);
        }

#ifdef DEBUG
        // face/eye ROIs
        rectangle( frame, faceROI,  Scalar( 255, 0, 0 ), 1, 8, 0 );
        rectangle( frame, eyeROI,  Scalar( 0, 255, 0 ), 1, 8, 0 );

        // Draw Eyes
        Rect eye;
        for( size_t j = 0; j < eyes.size(); j++ )
        {
               eye.x = eyes[j].x + eyeROI.x;
               eye.y = eyes[j].y + eyeROI.y;
               eye.width = eyes[j].width;
               eye.height = eyes[j].height;
               rectangle( frame, eye, Scalar( 255, 0, 255 ), 1, 8, 0 );
        }

        // Shift centers relative to eye
        eye_center_L.x += leftEye.x + eyeROI.x;
        eye_center_L.y += leftEye.y + eyeROI.y;
        eye_center_R.x += rightEye.x + eyeROI.x;
        eye_center_R.y += rightEye.y + eyeROI.y;

        // Draw Eye Center
        drawEyeCenter(eye_center_L);
        drawEyeCenter(eye_center_R);

        imshow("eye gaze",frame);
#endif


        // Update timer
        dur = (clock()-start) / (double)CLOCKS_PER_SEC;

        if(gaze_count == (15/3)) {
            circle(game_frame, gaze, 10, Scalar(0, 255, 124), 20);
            imshow( game_window, game_frame ); waitKey(1);
        }
        if(gaze_count == (15*2/3)) {
            circle(game_frame, gaze, 10, Scalar(0, 255, 0), 20);
            imshow( game_window, game_frame ); waitKey(1);
        }

    } while(gaze_count < 15. && waitKey(1)!='q');


    if(!mid_not_corners) {
        // Obtian average eye_center ratio for
        gazeCalibrationsLeftEye[index][0] /= gaze_count;
        gazeCalibrationsLeftEye[index][1] /= gaze_count;
        gazeCalibrationsRightEye[index][0] /= gaze_count;
        gazeCalibrationsRightEye[index][1] /= gaze_count;
    }

    return;
}

// Only for debugging
void drawEyeCenter(Point eye_center) {
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

void initKalmanFilter(Point &pp) {
    return;
}

void sleep_for(unsigned int frames) {
    unsigned int frame_cnt = 0;
    while(frame_cnt < frames) {
        if(!cap.read(frame)) { cerr<< "unable to read frame\n"; return; }
        frame_cnt++;
    }
}


// Visually display colored quadrants in game window
// Display associated eye center fractions, brighter colors are right eye
void checkCalibrations() {
    Point eye_center;
    Scalar colorL, colorR;
    int brite = 150;
    int dark = 200;

    // Clear Frame
    game_frame = Mat::zeros(height_screen, width_screen, CV_8UC3);
    displayOverlay(game_window, "Results\nTL: green  TR: blue\nBL: red    BR: yellow", -1);

    // Display Quadrants
    int side = 800;
    rectangle( game_frame, Rect(width_screen/2-side/2, height_screen/2-side/2, side/2, side/2),  Scalar( 0,255,0 ), 2, 8, 0 );
    rectangle( game_frame, Rect(width_screen/2,        height_screen/2-side/2, side/2, side/2),  Scalar( 255,0,0 ), 2, 8, 0 );
    rectangle( game_frame, Rect(width_screen/2-side/2, height_screen/2,        side/2, side/2),  Scalar( 0,0,255 ), 2, 8, 0 );
    rectangle( game_frame, Rect(width_screen/2,        height_screen/2,        side/2, side/2),  Scalar( 0,255,255 ), 2, 8, 0 );

    // Report Fractional Coordinates
    printf("Left Eye:\n");
    printf("TL: (%1.2f,%1.2f)\n", gazeCalibrationsLeftEye[0][0], gazeCalibrationsLeftEye[0][1]);
    printf("TR: (%1.2f,%1.2f)\n", gazeCalibrationsLeftEye[1][0], gazeCalibrationsLeftEye[1][1]);
    printf("BL: (%1.2f,%1.2f)\n", gazeCalibrationsLeftEye[2][0], gazeCalibrationsLeftEye[2][1]);
    printf("BR: (%1.2f,%1.2f)\n", gazeCalibrationsLeftEye[3][0], gazeCalibrationsLeftEye[3][1]);
    printf("Right Eye:\n");
    printf("TL: (%1.2f,%1.2f)\n", gazeCalibrationsRightEye[0][0], gazeCalibrationsRightEye[0][1]);
    printf("TR: (%1.2f,%1.2f)\n", gazeCalibrationsRightEye[1][0], gazeCalibrationsRightEye[1][1]);
    printf("BL: (%1.2f,%1.2f)\n", gazeCalibrationsRightEye[2][0], gazeCalibrationsRightEye[2][1]);
    printf("BR: (%1.2f,%1.2f)\n", gazeCalibrationsRightEye[3][0], gazeCalibrationsRightEye[3][1]);

    // Display Crosshairs of Eye Center Averages
    for(int i = 0; i<4; i++) {
        if(i==0) {
            colorL = Scalar(0,dark,0);
            colorR = Scalar(brite,255,brite);
        }
        else if(i==1) {
            colorL = Scalar(dark,0,0);
            colorR = Scalar(255,brite,brite);
        }
        else if(i==2) {
            colorL = Scalar(0,0,dark);
            colorR = Scalar(brite,brite,255);
        } else {
            colorL = Scalar(0,180,180);
            colorR = Scalar(brite,255,255);
        }

        // BL -> (0,0)

        // Left Eye
        eye_center.x = gazeCalibrationsLeftEye[i][0] * side + width_screen/2-side/2;
        eye_center.y = gazeCalibrationsLeftEye[i][1] * side + height_screen/2-side/2;
        crosshair1.x = eye_center.x;
        crosshair1.y = eye_center.y-5;
        crosshair2.x = eye_center.x;
        crosshair2.y = eye_center.y+5;
        line( game_frame, crosshair1, crosshair2, colorL, 1, 8, 0 );
        crosshair1.x = eye_center.x-5;
        crosshair1.y = eye_center.y;
        crosshair2.x = eye_center.x+5;
        crosshair2.y = eye_center.y;
        line( game_frame, crosshair1, crosshair2, colorL, 1, 8, 0 );

        // Right Eye
        eye_center.x = gazeCalibrationsRightEye[i][0] * side + width_screen/2-side/2;
        eye_center.y = gazeCalibrationsRightEye[i][1] * side + height_screen/2-side/2;
        crosshair1.x = eye_center.x;
        crosshair1.y = eye_center.y-5;
        crosshair2.x = eye_center.x;
        crosshair2.y = eye_center.y+5;
        line( game_frame, crosshair1, crosshair2, colorR, 1, 8, 0 );
        crosshair1.x = eye_center.x-5;
        crosshair1.y = eye_center.y;
        crosshair2.x = eye_center.x+5;
        crosshair2.y = eye_center.y;
        line( game_frame, crosshair1, crosshair2, colorR, 1, 8, 0 );
    }

    imshow( game_window, game_frame ); waitKey(1);
    waitForSpace();
}


// Visually display colored quadrants in game window
// Display associated eye center fractions, brighter colors are right eye
void checkCalibrationsMid() {
    Scalar colorL = Scalar(0,255,0);
    Scalar colorR = Scalar(0,0,255);

    // Clear Frame
    game_frame = Mat::zeros(height_screen, width_screen, CV_8UC3);
    displayOverlay(game_window, "Results\nLeft Eye: green  \nRight Eye: red", -1);

    // Display Quadrants
    int side = 800;
    rectangle( game_frame, Rect(width_screen/2-side/2, height_screen/2-side/2, side/2, side/2),  Scalar( 255,255,255 ), 2, 8, 0 );
    rectangle( game_frame, Rect(width_screen/2,        height_screen/2-side/2, side/2, side/2),  Scalar( 255,255,255 ), 2, 8, 0 );
    rectangle( game_frame, Rect(width_screen/2-side/2, height_screen/2,        side/2, side/2),  Scalar( 255,255,255 ), 2, 8, 0 );
    rectangle( game_frame, Rect(width_screen/2,        height_screen/2,        side/2, side/2),  Scalar( 255,255,255 ), 2, 8, 0 );

    // Report Fractional Coordinates
    printf("Left Eye:\n");
    printf("Middle: (%1.2f,%1.2f)\n", gazeCalibrationsLeftEye[4][0], gazeCalibrationsLeftEye[4][1]);
    printf("Right Eye:\n");
    printf("Middle: (%1.2f,%1.2f)\n", gazeCalibrationsRightEye[4][0], gazeCalibrationsRightEye[4][1]);

    // Display Crosshairs of Eye Center Averages

    // BL -> (0,0)

    rectangle( game_frame,
               Rect(-gazeCalibrationsLeftEye[4][0]*side + width_screen/2,
                    -gazeCalibrationsLeftEye[4][1]*side + height_screen/2,
                    2.*gazeCalibrationsLeftEye[4][0]*side,
                    2.*gazeCalibrationsLeftEye[4][1]*side),
              colorL, 1, 8, 0 );

    rectangle( game_frame,
               Rect(-gazeCalibrationsRightEye[4][0]*side  + width_screen/2,
                    -gazeCalibrationsRightEye[4][1]*side  + height_screen/2,
                    2.*gazeCalibrationsRightEye[4][0]*side,
                    2.*gazeCalibrationsRightEye[4][1]*side),
               colorR, 1, 8, 0 );

    imshow( game_window, game_frame ); waitKey(1);
    waitForSpace();
}

int startGame() {
    int your_score = 0;
    vector<Point> eye_centers(2,Point(0,0));
    Point eye_center_avg, eye1,eye2;
    Rect eyes_avg;
    vector<Rect> bounds_to_disp(2,Rect(0,0,0,0));
    double xBound = max(gazeCalibrationsLeftEye[4][0], gazeCalibrationsRightEye[4][0]);
    double yBound = max(gazeCalibrationsLeftEye[4][1], gazeCalibrationsRightEye[4][1]);

    xBound = 1.2*xBound;
    yBound = 1.2*yBound;

    // Prompt to start
    game_frame = Mat::zeros(height_screen, width_screen, CV_8UC3);
    displayOverlay(game_window, "Great!  Let the game begin! Press 'space' to start", -1);

    imshow( game_window, game_frame ); waitKey(1);
    waitForSpace();



    while((char)cc != 'q'){
        // Read Frame / Pre-Process
        if(!cap.read(frame)) { cerr<< "unable to read frame\n"; return -1; }
        // NOTE: eye center estimation REQUIRES grey scale
        flip(frame, frame, 1);
        eye_search_space = frame(eyeROI);
        cvtColor( eye_search_space, eye_search_space, COLOR_BGR2GRAY );

        // Find Eyes
        eyes_cascade.detectMultiScale(eye_search_space, eyes, 1.1, 2, 0 |CV_HAAR_SCALE_IMAGE, Size(50,50),Size(80,80));

        bounds_to_disp[0].width = 2.*xBound*eyes[0].width;
        bounds_to_disp[0].height = 2.*yBound*eyes[0].height;
        bounds_to_disp[0].x = eyes[0].x+(0.5-xBound)*eyes[0].width;
        bounds_to_disp[0].y = eyes[0].y+(0.5-yBound)*eyes[0].height;
        bounds_to_disp[1].width = 2*xBound*eyes[1].width;
        bounds_to_disp[1].height = 2*yBound*eyes[1].height;
        bounds_to_disp[1].x = eyes[1].x+(0.5-xBound)*eyes[1].width;
        bounds_to_disp[1].y = eyes[1].y+(0.5-yBound)*eyes[1].height;

          // Catch a blink
           if(eyes.size() == 0) {
               sprintf(overlay_str, "You Blinked!\nScore: %d\nPlay Again? (y/n)", score);
               break;
           }

           // Check Gaze
           if( eyes.size() == 2) {

               // Find Eye Center
               eye_centers[0] = findEyeCenter(eye_search_space, eyes[0], "_");
               eye_centers[1] = findEyeCenter(eye_search_space, eyes[1], "_");

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
               // Draw boundary region
               rectangle( eye_search_space, bounds_to_disp[0], Scalar(255,255,255), 1, 8, 0 );
               rectangle( eye_search_space, bounds_to_disp[1], Scalar(255,255,255), 1, 8, 0 );
               imshow( "Eye Search Space", eye_search_space ); cc=waitKey(1);

               // undo shift eye_centers
               eye_centers[0].x -= eyes[0].x;
               eye_centers[0].y -= eyes[0].y;
               eye_centers[1].x -= eyes[1].x;
               eye_centers[1].y -= eyes[1].y;

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

        your_score++;

        cc = waitKey(1);
    }


    displayOverlay(game_window, overlay_str, -1);
    imshow( game_window, game_frame ); waitKey(1);

    return your_score;
}

*/
