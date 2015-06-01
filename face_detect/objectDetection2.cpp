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
int startGame();
void sleep_for(unsigned int seconds);
void checkCalibrations();
void testFindEyeCenter();

void drawEyeCenter(Point eye_center);

/** Global variables */
String face_cascade_path = "/home/ee443/FinalProjectEE443/eye_tracker/src/face_detect/data/lbpcascades/";
String eye_cascade_path = "/home/ee443/FinalProjectEE443/eye_tracker/src/face_detect/data/haarcascades/";
//String eye_cascade_path = "/home/ee443/FinalProjectEE443/eye_tracker/src/face_detect/data/haarcascades_cuda/";

String face_cascade_name = "lbpcascade_frontalface.xml";
String eyes_cascade_name = "haarcascade_eye_tree_eyeglasses.xml";
//String eyes_cascade_name = "haarcascade_eye.xml";
//String eyes_cascade_name = "haarcascade_lefteye_2splits.xml";
//String eyes_cascade_name = "haarcascade_righteye_2splits.xml";

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
Rect leftEye, rightEye;

// Game
int width_screen = 1590;
int height_screen = 1100;
Mat game_frame;
Point gaze;

// TopLeft, TopRight, BottomLeft, BottomRight, Middle
double gazeCalibrationsLeftEye[5][2];
double gazeCalibrationsRightEye[5][2];

// Kalman objs
KalmanFilter RKF(4,2,0);
KalmanFilter LKF(4,2,0);

char overlay_str[100];
int score;



/**
 * @function main
 */
int main( void )
{
//  cap.open();
  cout.precision(numeric_limits<double>::digits10);

  // Load the cascade
  if( !face_cascade.load( face_cascade_path + face_cascade_name ) ){ printf("--(!)objectDetection2:Error loading face_cascade files\n"); return -1; };
  if( !eyes_cascade.load( eye_cascade_path + eyes_cascade_name ) ){ printf("--(!)objectDetection2:Error loading eyes_cascade files\n"); return -1; };


  // Initialize Game
  cout << "Game is initialized: " << boolalpha << initializeGame() << endl;


  // Play Game
  do {
    score = startGame();


    sprintf(overlay_str, "You Blinked!\nScore: %d\nPlay Again? (y/n)", score);
    displayOverlay(game_window, overlay_str, -1);
    imshow( game_window, game_frame ); waitKey(1);

    cc = 0;
    while((char)cc != 'y' && (char)cc != 'n') {
        cc = waitKey(1);
    }


  } while((char)cc != 'n');


  return 0;
}




// Reads videofeed, performs face detection, defines search space for eyes
bool initializeGame() {
    double widthCrop=0.75,heightCrop=0.4,CropYShift=0.18;
//    double widthCrop=1.,heightCrop=1.,CropYShift=0.;

    // Starting Screen
    game_frame = Mat::zeros(height_screen, width_screen, CV_8UC3);
    imshow( game_window, game_frame ); waitKey(1);
    displayOverlay(game_window, "Welcome to the Staring Game!\n\n Press 'space' to continue.", -1);
    waitForSpace();

    // Pre - Face Finding
    displayOverlay(game_window, "Okay we are going to try to find your face,\n so get comfortable because you can't move after this!\n\nPress 'space' to continue.", -1);
    imshow( game_window, game_frame ); waitKey(1);
    waitForSpace();
    int minFaceDim = 200; //300;

    // Face Finding
    while((char)cc == 'r' || faceROI.width < minFaceDim) {
        cc = 0;
        displayOverlay(game_window, "Working...", -1); imshow( game_window, game_frame ); waitKey(1);

        // Find Face ROI
        if(findFaceROI(1) < 0){ return false; }

        // Set eyeROI as restricted portion of faceROI
        eyeROI.width = widthCrop*faceROI.width;
        eyeROI.x = faceROI.x + (1.-widthCrop)/2*faceROI.width;
        eyeROI.height = heightCrop*faceROI.height;
        eyeROI.y = faceROI.y + CropYShift*faceROI.height;

        // Display user's image, face, and eye rois on game screen
        rectangle( frame, faceROI,  Scalar( 255, 0, 0 ), 1, 8, 0 );
        rectangle( frame, eyeROI,  Scalar( 0, 255, 0 ), 1, 8, 0 );
        frame.copyTo(game_frame(Rect(width_screen/2 - frame.cols/2, height_screen/2 - frame.rows/2, frame.cols, frame.rows)));

        // Verify roi is approproate
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
    double left_side_frac_LE, right_side_frac_LE, top_side_frac_LE, bot_side_frac_LE;
    double left_side_frac_RE, right_side_frac_RE, top_side_frac_RE, bot_side_frac_RE;
    Rect tempL = Rect(0,0,0,0);
    Rect tempR = Rect(0,0,0,0);
    do {
        // Pre - Gaze Calibration
        if(tempL.x == 0) {
            displayOverlay(game_window, "Great!  Now we are going to do some gaze calibrations.\nYou need to stare at the dot.\n\nPress 'space' to continue.", -1); imshow( game_window, game_frame ); waitKey(1);
        } else {
            displayOverlay(game_window, "Shoot!  Looks like we need to do that again.\n\nPress 'space' to continue.", -1); imshow( game_window, game_frame ); waitKey(1);
        }
        waitForSpace();

        // Artificial Calibration:
        gazeCalibrationsLeftEye[0][0] = .45;
        gazeCalibrationsLeftEye[0][1] = .45;
        gazeCalibrationsRightEye[0][0] = .46;
        gazeCalibrationsRightEye[0][1] = .45;
        gazeCalibrationsLeftEye[1][0] = .55;
        gazeCalibrationsLeftEye[1][1] = .45;
        gazeCalibrationsRightEye[1][0] = .56;
        gazeCalibrationsRightEye[1][1] = .45;
        gazeCalibrationsLeftEye[2][0] = .45;
        gazeCalibrationsLeftEye[2][1] = .55;
        gazeCalibrationsRightEye[2][0] = .46;
        gazeCalibrationsRightEye[2][1] = .55;
        gazeCalibrationsLeftEye[3][0] = .55;
        gazeCalibrationsLeftEye[3][1] = .55;
        gazeCalibrationsRightEye[3][0] = .56;
        gazeCalibrationsRightEye[3][1] = .55;

        // Calibrate Gaze
//        if(calibrate() < 0){ return false; }

        left_side_frac_LE = (gazeCalibrationsLeftEye[0][0] + gazeCalibrationsLeftEye[2][0])/2.;
        right_side_frac_LE = (gazeCalibrationsLeftEye[1][0] + gazeCalibrationsLeftEye[3][0])/2.;
        top_side_frac_LE = (gazeCalibrationsLeftEye[0][1] + gazeCalibrationsLeftEye[1][1])/2.;
        bot_side_frac_LE = (gazeCalibrationsLeftEye[2][1] + gazeCalibrationsLeftEye[3][1])/2.;

        left_side_frac_RE = (gazeCalibrationsRightEye[0][0] + gazeCalibrationsRightEye[2][0])/2.;
        right_side_frac_RE = (gazeCalibrationsRightEye[1][0] + gazeCalibrationsRightEye[3][0])/2.;
        top_side_frac_RE = (gazeCalibrationsRightEye[0][1] + gazeCalibrationsRightEye[1][1])/2.;
        bot_side_frac_RE = (gazeCalibrationsRightEye[2][1] + gazeCalibrationsRightEye[3][1])/2.;

//#ifdef DEBUG
        // Check Calibrations
        checkCalibrations();

        int side = 800;
        tempL.x = width_screen/2-side/2 + left_side_frac_LE * side;
        tempL.y = height_screen/2-side/2 + top_side_frac_LE * side;
        tempL.width = width_screen/2-side/2 + (right_side_frac_LE-left_side_frac_LE) * side;
        tempL.height = height_screen/2-side/2 + (top_side_frac_LE-bot_side_frac_LE) * side;

        rectangle( game_frame, tempL,  Scalar( 255, 255, 255 ), 1, 8, 0 );
//#endif

//    } while(left_side_frac_LE > right_side_frac_LE || bot_side_frac_LE > top_side_frac_LE);
    } while(false);

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

// Update gazeCalibrations to have fractional (x,y) coordinates
// corresponding to staring at location x,y on the screen
void getGaze(int x, int y, int index) {
    double gaze_count = 0.;
    Point eye_center_L, eye_center_R;
    clock_t start;
    double dur, border, fraction[4]; //fr[0:2) --> L(x/W,y/H); fr[2:4) --> R(x/W,y/H)

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

            // Find Eye Center Fractions
            fraction[0] = (double)eye_center_L.x / (double)leftEye.width;
            fraction[1] = (double)eye_center_L.y / (double)leftEye.height;
            fraction[2] = (double)eye_center_R.x / (double)rightEye.width;
            fraction[3] = (double)eye_center_R.y / (double)rightEye.height;

            // Check center is not excessively close to ROI border (eyebrows or something)
            border = 0.25;
            if(fraction[0] < border || fraction[0] > (1.-border) ||
                    fraction[1] < border || fraction[1] > (1.-border) ||
                    fraction[2] < border || fraction[2] > (1.-border) ||
                    fraction[3] < border || fraction[3] > (1.-border)) {
                continue;
            }

            // Accumulate eye_center ratios for each eye
            // Update gazeCalibrationsLeftEye
            gazeCalibrationsLeftEye[index][0] += fraction[0];
            gazeCalibrationsLeftEye[index][1] += fraction[1];
            // Update gazeCalibrationsRightEye
            gazeCalibrationsRightEye[index][0] += fraction[2];
            gazeCalibrationsRightEye[index][1] += fraction[3];

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

//        if(eyes.size() == 2) {
//            char str[16];
//            sprintf(str, "%d", (int)(1000000*x + 1000*y + gaze_count/2));
//            imshow(str,frame);
//        }
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

    // Obtian average eye_center ratio for
    gazeCalibrationsLeftEye[index][0] /= gaze_count;
    gazeCalibrationsLeftEye[index][1] /= gaze_count;
    gazeCalibrationsRightEye[index][0] /= gaze_count;
    gazeCalibrationsRightEye[index][1] /= gaze_count;

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


/*
bool testFindEyeCenter(VideoCapture &cap) {
    Mat roi;
    Rect face;
    clock_t ss;
    double dd;


    do
    {
        if(cap.isOpened())
        {
            if(cap.read(frame))
            {
                flip(frame, frame, 1);

                // Find Face
                if(detectFace(face) == 0) {
                    // Find Eyes
                    roi = frame( face );
                    eyes_cascade.detectMultiScale(roi, eyes, 1.1, 2, 0 |CV_HAAR_SCALE_IMAGE, Size(50,50),Size(80,80));

                    ss = clock();
                    // Find Left Eye Center
                    eye_center_L = findEyeCenter(roi, leftEye, "_");
                    dd = (clock()-ss) / (double)CLOCKS_PER_SEC;


                    rectangle( frame, roi,  Scalar( 255, 0, 0 ), 1, 8, 0 );
                    rectangle( frame, eyes[0],  Scalar( 0, 255, 0 ), 1, 8, 0 );
                    rectangle( frame, eyes[1],  Scalar( 0, 255, 0 ), 1, 8, 0 );
                }
            }
            else { cerr << " --(!) Could not read frame\n" <<endl; return false; }
        }
        else{ cerr << " --(!) Could not open videostream\n" <<endl; return false; }



    } while(waitKey(1)!='q');

    return true;
}
*/




int startGame() {
    game_frame = Mat::zeros(height_screen, width_screen, CV_8UC3);
    displayOverlay(game_window, "Great!  Let the game begin! Press 'space' to start", -1);
    imshow( game_window, game_frame ); waitKey(1);
    waitForSpace();

    char str[50];
    int your_score = 0;

    while((char)cc != 'q'){
        // Read Frame / Pre-Process
        if(!cap.read(frame)) { cerr<< "unable to read frame\n"; return -1; }
        flip(frame, frame, 1);
//        cvtColor( frame, frame, COLOR_BGR2GRAY );
        eye_search_space = frame(eyeROI);


        // Find Eyes
        eyes_cascade.detectMultiScale(eye_search_space, eyes, 1.1, 2, 0 |CV_HAAR_SCALE_IMAGE, Size(50,50),Size(80,80));

        // Catch a blink
        if(eyes.size() == 0) {
            break;
        }

        // Draw Eyes and Display
        for( size_t j = 0; j < eyes.size(); j++ )
        {
            rectangle( eye_search_space, eyes[j], Scalar( 255, 0, 255 ), 1, 8, 0 );
        }
        imshow( "Eye Search Space", eye_search_space );

        your_score++;

        cc = waitKey(1);
    }

    return your_score;

}
