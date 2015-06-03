// one grande file
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <stdio.h>
#include <sstream>
#include <string>


#include "globals.h"
#include "helpers.h"


using namespace std;
using namespace cv;


cv::CascadeClassifier face_cascade;
cv::VideoCapture capture(0);


int detectFace(Mat &frame, Rect &face);

// Performs face detection
// Defines search space for eyes
//
bool initializeGame(Mat &game_frame, Rect &eyeROI) {
    //__________ADJUSTABLE_PARAMS_______________
    //
    //
    // Portion of face to crop for eyeROI
    double widthCrop=0.75,heightCrop=0.4,CropYShift=0.18;

    // Restrict how close user is to screen
    int minFaceDim = 200; //300;
    int maxFaceDim = 500;

    //__________DECLARATIONS_____________________
    //
    //
    int cc;        // for keyboard input
    int face_count, num_frames;
    Rect faceROI, face_avg, face;
    Mat frame;

    //__________START_SCREEN_____________________
    //
    //
    game_frame = Mat::zeros(height_screen, width_screen, CV_8UC3);
    imshow( game_window, game_frame ); waitKey(1);
    displayOverlay(game_window, "Welcome to the Staring Game!\n\n Press 'space' to continue.", -1);
    waitForSpace();

    //__________FACE_CODE________________________
    //
    //
    // Load the cascade
    if( !face_cascade.load( face_cascade_path + face_cascade_name ) ){ printf("--(!)objectDetection2:Error loading face_cascade files\n"); return -1; };

    displayOverlay(game_window, "Okay we are going to try to find your face,\n so get comfortable because you can't move after this!\n\nPress 'space' to continue.", -1);
    imshow( game_window, game_frame ); waitKey(1);
    waitForSpace();

    // Face Finding
    while((char)cc == 'r' || faceROI.width < minFaceDim || faceROI.width > maxFaceDim) {
        cc = 0;
        displayOverlay(game_window, "Working...", -1); imshow( game_window, game_frame ); waitKey(1);

        // Find Face ROI
        face_count = 0;
        num_frames = 10;
        do
        {
            if(capture.isOpened())
            {
                if(capture.read(frame))
                {
                    flip(frame, frame, 1);

                    // Find Face
                    if(detectFace(frame,face) == 0) {
                        // Add detected Rect to average
                        face_avg.width += face.width;
                        face_avg.height += face.height;
                        face_avg.x += face.x;
                        face_avg.y += face.y;
                        face_count++;
                    }
                }
                else { cerr << " --(!) Could not read frame\n" <<endl; return false; }
            }
            else{ cerr << " --(!) Could not open videostream\n" <<endl; return false; }

        } while(face_count<num_frames && waitKey(1)!='q');

        faceROI.width = face_avg.width/face_count;
        faceROI.height = face_avg.height/face_count;
        faceROI.x = face_avg.x/face_count;
        faceROI.y = face_avg.y/face_count;

        // Set eyeROI as restricted portion of faceROI
        eyeROI.width = widthCrop*faceROI.width;
        eyeROI.x = faceROI.x + (1.-widthCrop)/2*faceROI.width;
        eyeROI.height = heightCrop*faceROI.height;
        eyeROI.y = faceROI.y + CropYShift*faceROI.height;

        // Display user's image, face, and eye ROIs on game screen
        rectangle( frame, faceROI,  Scalar( 255, 0, 0 ), 1, 8, 0 );
        rectangle( frame, eyeROI,  Scalar( 0, 255, 0 ), 1, 8, 0 );
        frame.copyTo(game_frame(Rect(width_screen/2 - frame.cols/2, height_screen/2 - frame.rows/2, frame.cols, frame.rows)));

        // Verify ROI is approproate
        if(faceROI.width < minFaceDim) {
            displayOverlay(game_window, "Get closer to the screen!\n\nPress 'space' to try again.", -1);
        } else
        if(faceROI.width > maxFaceDim) {
            displayOverlay(game_window, "You're too close to the screen!\n\nPress 'space' to try again.", -1);
        } else {
            displayOverlay(game_window, "How does this look?\n(The location of the box, we know you're beautiful)\nMake sure that both eyes are clearly in the green box.\n\nPress 'space' to continue, 'r' to try again.", -1);
        }
        imshow( game_window, game_frame ); waitKey(1);

        #ifdef DEBUG
            // Print Face & Eye ROI Dimensions
            cout << "FACE  w: " << faceROI.width << " h: " << faceROI.height << "\n";
            cout << "EYES  w: " << eyeROI.width << " h: " << eyeROI.height << "\n";
        #endif

        while((char)cc != ' ' && (char)cc != 'r') {
            cc = waitKey(1);
        }
    }
    cc = 0;




//    //__________CENTER_CALIBRATION___________________
//    //
//    //
//    double left_side_frac_LE, right_side_frac_LE, top_side_frac_LE, bot_side_frac_LE;
//    double left_side_frac_RE, right_side_frac_RE, top_side_frac_RE, bot_side_frac_RE;
//    Rect tempL = Rect(0,0,0,0);
//    Rect tempR = Rect(0,0,0,0);
//    do {
//        // Pre - Gaze Calibration
//        if(tempL.x == 0) {
//            displayOverlay(game_window, "Great!  Now we are going to do some gaze calibrations.\nYou need to stare at the dot.\n\nPress 'space' to continue.", -1); imshow( game_window, game_frame ); waitKey(1);
//        } else {
//            displayOverlay(game_window, "Shoot!  Looks like we need to do that again.\n\nPress 'space' to continue.", -1); imshow( game_window, game_frame ); waitKey(1);
//        }
//        waitForSpace();

//        // Calibrate Gaze
//        if(calibrate() < 0){ return false; }

//        left_side_frac_LE = (gazeCalibrationsLeftEye[0][0] + gazeCalibrationsLeftEye[2][0])/2.;
//        right_side_frac_LE = (gazeCalibrationsLeftEye[1][0] + gazeCalibrationsLeftEye[3][0])/2.;
//        top_side_frac_LE = (gazeCalibrationsLeftEye[0][1] + gazeCalibrationsLeftEye[1][1])/2.;
//        bot_side_frac_LE = (gazeCalibrationsLeftEye[2][1] + gazeCalibrationsLeftEye[3][1])/2.;

//        left_side_frac_RE = (gazeCalibrationsRightEye[0][0] + gazeCalibrationsRightEye[2][0])/2.;
//        right_side_frac_RE = (gazeCalibrationsRightEye[1][0] + gazeCalibrationsRightEye[3][0])/2.;
//        top_side_frac_RE = (gazeCalibrationsRightEye[0][1] + gazeCalibrationsRightEye[1][1])/2.;
//        bot_side_frac_RE = (gazeCalibrationsRightEye[2][1] + gazeCalibrationsRightEye[3][1])/2.;

//////#ifdef DEBUG
//        // Check Calibrations
//        if(mid_not_corners) {
//            checkCalibrationsMid();
//        } else {
//            checkCalibrations();
//        }

//        int side = 800;
//        tempL.x = width_screen/2-side/2 + left_side_frac_LE * side;
//        tempL.y = height_screen/2-side/2 + top_side_frac_LE * side;
//        tempL.width = width_screen/2-side/2 + (right_side_frac_LE-left_side_frac_LE) * side;
//        tempL.height = height_screen/2-side/2 + (top_side_frac_LE-bot_side_frac_LE) * side;

//        rectangle( game_frame, tempL,  Scalar( 255, 255, 255 ), 1, 8, 0 );
////#endif

////    } while(left_side_frac_LE > right_side_frac_LE || bot_side_frac_LE > top_side_frac_LE);
//    } while(false);

//    // eyeROI

    return true;
}









// Used in findFaceROI()
int detectFace(Mat &frame, Rect &face) {
    Mat frame_gray;
    vector<Rect> faces;

    //-- Pre-process frame
    cvtColor( frame, frame_gray, COLOR_BGR2GRAY );
    equalizeHist( frame_gray, frame_gray ); // maybe remove this for speed if unnecessary for quality

    //-- Detect faces
    face_cascade.detectMultiScale( frame_gray, faces, 1.1, 2, 0, Size(150, 150),Size(400,400) );


    // for now, assume only one face is in frame
    // later, if mult faces are in frame, then take face closest to center of frame

    if(faces.size()==0){ face = Rect(); return -1; } //no face was found
    face = faces[0]; return 0;
}

