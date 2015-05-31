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

#define DEBUG

using namespace std;
using namespace cv;

/** Function Headers */
bool initializeGame();
bool findFaceROI(int averaging_duration);
int calibrate();
int detectFace( Rect &face );
void getGaze(int x, int y, int index);
void waitForSpace();
void initKalmanFilter(Point&);
void startGame();
void sleep_for(unsigned int seconds);
void checkCalibrations();

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
double gazeCalibrations[5][2];

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

  // Load the cascade
  if( !face_cascade.load( cascade_path + face_cascade_name ) ){ printf("--(!)objectDetection2:Error loading face_cascade files\n"); return -1; };
  if( !eyes_cascade.load( cascade_path + eyes_cascade_name ) ){ printf("--(!)objectDetection2:Error loading eyes_cascade files\n"); return -1; };

  // Initialize Game
  cout << "Game is initialized: " << boolalpha << initializeGame() << endl;


  startGame();

  return 0;
}




// Reads videofeed, performs face detection, defines search space for eyes
bool initializeGame() {
    // Starting Screen
    game_frame = Mat::zeros(height_screen, width_screen, CV_8UC3);
    imshow( game_window, game_frame ); waitKey(1);
    displayOverlay(game_window, "Welcome to the Staring Game!\n\n Press 'space' to continue.", -1);
    waitForSpace();

    // Pre - Face Finding
    displayOverlay(game_window, "Okay we are going to try to find your face,\n so get comfortable because you can't move after this!\n\nPress 'space' to continue.", -1);
    imshow( game_window, game_frame ); waitKey(1);
    waitForSpace();
    int minFaceDim = 300;

    // Face Finding
    while((char)cc == 'r' || faceROI.width < minFaceDim) {
        cc = 0;
        displayOverlay(game_window, "Working...", -1); imshow( game_window, game_frame ); waitKey(1);

        // Find Face ROI
        if(findFaceROI(1) < 0){ return false; }

        // Set eyeROI as restricted portion of faceROI
        double widthCrop = 0.75;
        eyeROI.width = widthCrop*faceROI.width;
        eyeROI.x = faceROI.x + (1.-widthCrop)/2*faceROI.width;
        eyeROI.height = 0.4*faceROI.height;
        eyeROI.y = faceROI.y + 0.18*faceROI.height;

        rectangle( frame, faceROI,  Scalar( 255, 0, 0 ), 1, 8, 0 );
        rectangle( frame, eyeROI,  Scalar( 0, 255, 0 ), 1, 8, 0 );
        frame.copyTo(game_frame(Rect(width_screen/2 - frame.cols/2, height_screen/2 - frame.rows/2, frame.cols, frame.rows)));

        if(faceROI.width < minFaceDim) {
            displayOverlay(game_window, "Get closer to the screen!\n\nPress 'space' to try again.", -1);
        } else {
            displayOverlay(game_window, "How does this look?\n(The location of the box, we know you're beautiful)\nMake sure that both eyes are clearly in the green box.\n\nPress 'space' to continue, 'r' to try again.", -1);
        }
        imshow( game_window, game_frame ); waitKey(1);

        cout << "FACE  w: " << faceROI.width << " h: " << faceROI.height << "\n";
        cout << "EYES  w: " << eyeROI.width << " h: " << eyeROI.height << "\n";

        while((char)cc != ' ' && (char)cc != 'r') {
            cc = waitKey(1);
        }
    }
    cc = 0;

    // Calibrating
    double left_frac, right_frac, top_frac, bot_frac;
    Rect temp = Rect(0,0,0,0);
    do {
        // Pre - Gaze Calibration
        if(temp.x == 0) {
            displayOverlay(game_window, "Great!  Now we are going to do some gaze calibrations.\nYou need to stare at the dot.\n\nPress 'space' to continue.", -1); imshow( game_window, game_frame ); waitKey(1);
        } else {
            displayOverlay(game_window, "Shoot!  Looks like we need to do that again.\n\nPress 'space' to continue.", -1); imshow( game_window, game_frame ); waitKey(1);
        }
        waitForSpace();

        // Calibrate Gaze
        if(calibrate() < 0){ return false; }

        left_frac = (gazeCalibrations[0][0] + gazeCalibrations[2][0])/2.;
        right_frac = (gazeCalibrations[1][0] + gazeCalibrations[3][0])/2.;
        top_frac = (gazeCalibrations[0][1] + gazeCalibrations[1][1])/2.;
        bot_frac = (gazeCalibrations[2][1] + gazeCalibrations[3][1])/2.;

//#ifdef DEBUG
        // Check Calibrations
        checkCalibrations();

        int side = 400;
        temp.x = width_screen/2-side/2 + left_frac * side;
        temp.y = height_screen/2-side/2 + top_frac * side;
        temp.width = width_screen/2-side/2 + (right_frac-left_frac) * side;
        temp.height = height_screen/2-side/2 + (top_frac-bot_frac) * side;

        rectangle( game_frame, temp,  Scalar( 255, 255, 255 ), 1, 8, 0 );
//#endif

    } while(left_frac > right_frac || bot_frac > top_frac);

    startGame();

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
bool findFaceROI(int averaging_duration) {
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
                if(detectFace(face) == 0) {
                    // Add detected Rect to average
                    face_avg.width += face.width; face_avg.height += face.height;
                    face_avg.x += face.x; face_avg.y += face.y;
                    face_count++;
                }
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

    return true;
}

// Used in findFaceROI()
int detectFace(Rect &face) {
    cvtColor( frame, frame_gray, COLOR_BGR2GRAY );

    equalizeHist( frame_gray, frame_gray ); // maybe remove this for speed if unnecessary for quality

    //-- Detect faces
    face_cascade.detectMultiScale( frame_gray, faces, 1.1, 2, 0, Size(150, 150),Size(400,400) );


    // for now, assume only one face is in frame
    // later, if mult faces are in frame, then take face closest to center of frame

    if(faces.size()==0){ face = Rect(); return -1; } //no face was found
    face = faces[0]; return 0;
}

// initializes gaze calibration points for screen corners and center
int calibrate() {
    // Perform calibration
    getGaze(20,50,0);
    getGaze(width_screen-20,50,1);
    getGaze(20,height_screen-20,2);
    getGaze(width_screen-20,height_screen-20,3);
//    gazeCalibrations[4] = getGaze(width_screen/2,height_screen/2);

    return 0;
}

// Update gazeCalibrations[index] to have fractional (x,y) coordinates
// corresponding to staring at location x,y on the screen
void getGaze(int x, int y, int index) {
    double gaze_count = 0.;
    Point eye_center;
    clock_t start;
    double dur;

    // Display the circle
    game_frame = Mat::zeros(height_screen, width_screen, CV_8UC3);
    if(x >= 0) {
        gaze.x = x;
        gaze.y = y;
        circle(game_frame, gaze, 10, Scalar(0, 0, 255), 20);
    }
    displayOverlay(game_window, "", -1); imshow( game_window, game_frame ); waitKey(1);
    sleep_for(50);

    circle(game_frame, gaze, 10, Scalar(0, 255, 0), 20);
    imshow( game_window, game_frame ); waitKey(1);

    // Initialize gaze value
    gazeCalibrations[index][0] = 0.;
    gazeCalibrations[index][1] = 0.;

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
                eye_center = findEyeCenter(eye_search_space, eyes[j], "_");

                // Update gazeCalibrations[index]
                gazeCalibrations[index][0] += (double)eye_center.x / (double)eyes[j].width;
                gazeCalibrations[index][1] += (double)eye_center.y / (double)eyes[j].height;
                gaze_count++;



#ifdef DEBUG
                cout << "(" << (double)eye_center.x / (double)eyes[j].width << "," << (double)eye_center.y / (double)eyes[j].height << ")\n";

                // Shift relative to eye
                eye_center.x += eyes[j].x + eyeROI.x;
                eye_center.y += eyes[j].y + eyeROI.y;

                // Draw Eye Center
                // (only one eye! put in "for each eye" loop to see both)
                drawEyeCenter(eye_center);
#endif
            }
        }

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
        }

//        imshow("eye gaze",frame);


        if(eyes.size() == 2) {
            char str[16];
            sprintf(str, "%d", (int)(1000000*x + 1000*y + gaze_count/2));
            imshow(str,frame);
        }
#endif


        // Update timer
        dur = (clock()-start) / (double)CLOCKS_PER_SEC;

    } while(gaze_count < 2 * 10. && waitKey(1)!='q');


    gazeCalibrations[index][0] /= gaze_count;
    gazeCalibrations[index][1] /= gaze_count;

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



void checkCalibrations() {
    game_frame = Mat::zeros(height_screen, width_screen, CV_8UC3);
    displayOverlay(game_window, "Results\nTL: green  TR: blue\nBL: red    BR: yellow", -1);

    int side = 400;
    rectangle( game_frame, Rect(width_screen/2-side/2, height_screen/2-side/2, side/2, side/2),  Scalar( 0,255,0 ), 2, 8, 0 );
    rectangle( game_frame, Rect(width_screen/2,        height_screen/2-side/2, side/2, side/2),  Scalar( 255,0,0 ), 2, 8, 0 );
    rectangle( game_frame, Rect(width_screen/2-side/2, height_screen/2,        side/2, side/2),  Scalar( 0,0,255 ), 2, 8, 0 );
    rectangle( game_frame, Rect(width_screen/2,        height_screen/2,        side/2, side/2),  Scalar( 0,255,255 ), 2, 8, 0 );

    for(int i = 0; i<4; i++) {
        Point eye_center;
        Scalar color;

        if(i==0) {
            color = Scalar(0,255,0);
            printf("TL: (%1.2f,%1.2f)\n", gazeCalibrations[i][0], gazeCalibrations[i][1]);
        }
        else if(i==1) {
            color = Scalar(255,0,0);
            printf("TR: (%1.2f,%1.2f)\n", gazeCalibrations[i][0], gazeCalibrations[i][1]);
        }
        else if(i==2) {
            color = Scalar(0,0,255);
            printf("BL: (%1.2f,%1.2f)\n", gazeCalibrations[i][0], gazeCalibrations[i][1]);
        } else {
            color = Scalar(0,255,255);
            printf("BR: (%1.2f,%1.2f)\n", gazeCalibrations[i][0], gazeCalibrations[i][1]);
        }

        // BL -> (0,0)
        eye_center.x = gazeCalibrations[i][0] * side + width_screen/2-side/2;
        eye_center.y = gazeCalibrations[i][1] * side + height_screen/2-side/2;

        crosshair1.x = eye_center.x;
        crosshair1.y = eye_center.y-5;
        crosshair2.x = eye_center.x;
        crosshair2.y = eye_center.y+5;
        line( game_frame, crosshair1, crosshair2, color, 1, 8, 0 );
        crosshair1.x = eye_center.x-5;
        crosshair1.y = eye_center.y;
        crosshair2.x = eye_center.x+5;
        crosshair2.y = eye_center.y;
        line( game_frame, crosshair1, crosshair2, color, 1, 8, 0 );
    }

    imshow( game_window, game_frame ); waitKey(1);
    waitForSpace();
}


void startGame() {
    printf("Game Started!\n");

    displayOverlay(game_window, "Great!  Let the game begin!", -1); imshow( game_window, game_frame ); waitKey(1);


}




