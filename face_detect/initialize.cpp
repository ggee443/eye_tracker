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

#include "initialize.h"
#include "globals.h"
#include "helpers.h"
#include "findEyeCenter.h"

#define DEBUG

using namespace std;
using namespace cv;

cv::CascadeClassifier face_cascade;
cv::CascadeClassifier eyes_cascade;
cv::VideoCapture capture(0);


Mat frame, eye_search_space;
vector<Rect> eyes;
Rect faceROI;
Point crosshair1, crosshair2;
// from object_detection
extern double gazeCalibrations[2];

int detectFace(Rect &face);
void getGaze(Mat &game_frame, Rect &eyeROI, double* gazeCalibrations);
void drawEyeCenter(Point eye_center);
void checkCalibrationsMid(Mat &);

// Performs face detection
// Defines search space for eyes
//
bool initializeGame(cv::Mat &game_frame, cv::Rect &eyeROI, double* gazeCalibrations) {
    //__________ADJUSTABLE_PARAMS_______________
    //
    //
    // Portion of face to crop for eyeROI
    double widthCrop=0.75,heightCrop=0.4,CropYShift=0.18;

    // Restrict how close user is to screen
    int minFaceDim = 100; //300;
    int maxFaceDim = 800;

    //__________DECLARATIONS_____________________
    //
    //
    int cc;        // for keyboard input
    int face_count, num_frames;
    Rect face_avg, face;

    // Load the cascade classifiers
    if( !face_cascade.load( face_cascade_path + face_cascade_name ) ){ printf("--(!)objectDetection2:Error loading face_cascade files\n"); return -1; };
    if( !eyes_cascade.load( eye_cascade_path + eyes_cascade_name ) ){ printf("--(!)objectDetection2:Error loading eyes_cascade files\n"); return -1; };

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
        face_avg = Rect(0,0,0,0);
        do
        {
            if(capture.isOpened())
            {
                if(capture.read(frame))
                {
                    flip(frame, frame, 1);

                    // Find Face
                    if(detectFace(face) == 0) {
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




    //__________CENTER_CALIBRATION___________________
    //
    //

    do {
        cc = 0;

        // Pre - Gaze Calibration
        displayOverlay(game_window, "Great!  Now we are going to do some gaze calibrations.\nYou need to stare at the dot.\n\nPress 'space' to continue.", -1);
        imshow( game_window, game_frame ); cc = waitKey(1);
        waitForSpace();

        // Calibrate Gaze
        getGaze(game_frame,eyeROI, gazeCalibrations);

        // Check Calibrations
        checkCalibrationsMid(game_frame);

        // Prompt
        displayOverlay(game_window, "How Does that look?\n\nPress 'space' to continue, 'r' to try again.", -1);
        imshow( game_window, game_frame ); waitKey(1);

        while((char)cc != ' ' && (char)cc != 'r') {
            cc = waitKey(1);
        }

    } while(cc != ' ');

    return true;
}


// Only for debugging
void drawEyeCenter(Point eye_center) {
    // Draw Eye Center
    crosshair1.x = eye_center.x;
    crosshair1.y = eye_center.y-5;
    crosshair2.x = eye_center.x;
    crosshair2.y = eye_center.y+5;
    line( frame, crosshair1, crosshair2, Scalar( 255, 255, 255 ), 1, 8, 0 );
    crosshair1.x = eye_center.x-5;
    crosshair1.y = eye_center.y;
    crosshair2.x = eye_center.x+5;
    crosshair2.y = eye_center.y;
    line( frame, crosshair1, crosshair2, Scalar( 255, 255, 255 ), 1, 8, 0 );
}

// Used in findFaceROI()
int detectFace(Rect &face) {
    Mat frame_gray;
    vector<Rect> faces;
    double euclid1,euclid2;

    //-- Pre-process frame
    cvtColor( frame, frame_gray, COLOR_BGR2GRAY );
    equalizeHist( frame_gray, frame_gray ); // maybe remove this for speed if unnecessary for quality

    //-- Detect faces
    face_cascade.detectMultiScale( frame_gray, faces, 1.1, 2, 0, Size(150, 150),Size(400,400) );


    if(faces.size()==0){ face = Rect(); return -1; } //no face was found


    face = faces[0];
    // for now, assume only one face is in frame
    // later, if mult faces are in frame, then take face closest to center of frame
    if(faces.size()>1)
    {
        for(int ff=1;ff<faces.size();ff++)
        {
            euclid1 = abs(face.x+(0.5*face.width) - 0.5*frame.cols)+abs(face.y+(0.5*face.height) - 0.5*frame.cols);
            euclid2 = abs(faces[ff].x+(0.5*faces[ff].width) - 0.5*frame.cols)+abs(faces[ff].y+(0.5*faces[ff].height) - 0.5*frame.cols);
            if(euclid1 > euclid2){ face = faces[ff]; }
        }
    }
    return 0;
}

// Visually display colored quadrants in game window
// Display associated eye center fractions, brighter colors are right eye
void checkCalibrationsMid(Mat &game_frame) {

    // Clear Frame
    game_frame = Mat::zeros(height_screen, width_screen, CV_8UC3);
    displayOverlay(game_window, "Maximum Eye Center Deviation", -1);

    // Display Quadrants
    int side = 800;
    rectangle( game_frame, Rect(width_screen/2-side/2, height_screen/2-side/2, side/2, side/2),  Scalar( 255,255,255 ), 2, 8, 0 );
    rectangle( game_frame, Rect(width_screen/2,        height_screen/2-side/2, side/2, side/2),  Scalar( 255,255,255 ), 2, 8, 0 );
    rectangle( game_frame, Rect(width_screen/2-side/2, height_screen/2,        side/2, side/2),  Scalar( 255,255,255 ), 2, 8, 0 );
    rectangle( game_frame, Rect(width_screen/2,        height_screen/2,        side/2, side/2),  Scalar( 255,255,255 ), 2, 8, 0 );

    // Report Fractional Coordinates
    printf("Middle: (%1.2f,%1.2f)\n", gazeCalibrations[0], gazeCalibrations[1]);

    // BL -> (0,0)

    rectangle( game_frame,
               Rect(-gazeCalibrations[0]*side + width_screen/2,
                    -gazeCalibrations[1]*side + height_screen/2,
                    2.*gazeCalibrations[0]*side,
                    2.*gazeCalibrations[1]*side),
              Scalar(0,255,0), 1, 8, 0 );
}

// Update gazeCalibrations to have fractional (x,y) coordinates
// corresponding to staring at location x,y on the screen
void getGaze(Mat &game_frame, Rect &eyeROI, double* gazeCalibrations)
{
    double height_frac = 0.35;
    int gaze_count = 0,num_frames=15, frame_count=0;
    Point gaze;
    Point eye_centers[2];
    double fraction[4]; // ratio b/w eye center and eye dims

    // Display the circle
    game_frame = Mat::zeros(height_screen, width_screen, CV_8UC3);
    gaze.x = width_screen/2;
    gaze.y = height_screen * height_frac;
    circle(game_frame, gaze, 10, Scalar(0, 0, 255), 20);
    displayOverlay(game_window, "", -1); imshow( game_window, game_frame ); waitKey(1);
//  go to sleep
    while(frame_count < 10)
    {
        if(!capture.read(frame)) { cerr<< "unable to read frame\n"; return; }
        frame_count++;
    }

    circle(game_frame, gaze, 10, Scalar(0, 255, 255), 20);
    imshow( game_window, game_frame ); waitKey(1);

    // Initialize gaze value
    gazeCalibrations[0] = 0.;
    gazeCalibrations[1] = 0.;

    // detect eye centers and average them
    do {
        // Read frame
        if(!capture.read(frame)) { cerr<< "unable to read frame\n"; return; }
        flip(frame, frame, 1);
        eye_search_space = frame(eyeROI);
        cvtColor( eye_search_space, eye_search_space, COLOR_BGR2GRAY );

        // Find Eyes
        eyes_cascade.detectMultiScale(eye_search_space, eyes, 1.1, 2, 0 |CV_HAAR_SCALE_IMAGE, Size(50,50),Size(80,80));

        if( eyes.size() == 2) {

            // Find Left Eye Center
            eye_centers[0] = findEyeCenter(eye_search_space, eyes[0], "_");
            // Find Right Eye Center
            eye_centers[1] = findEyeCenter(eye_search_space, eyes[1], "_");

            printf("calibration - l_eye_center: (%d,%d); eye_box_dims: %dx%d\n", \
                   eye_centers[0].x,\
                   eye_centers[0].y,\
                   eyes[0].width,\
                   eyes[0].height);


            // Find Eye Center Fractions
            fraction[0] = (double)eye_centers[0].x / (double)eyes[0].width;
            fraction[1] = (double)eye_centers[0].y / (double)eyes[0].height;
            fraction[2] = (double)eye_centers[1].x / (double)eyes[1].width;
            fraction[3] = (double)eye_centers[1].y / (double)eyes[1].height;

            // Check center is not excessively close to ROI border (eyebrows or something)
            double border = 0.25;
            if(fraction[0] < border || fraction[0] > (1.-border) ||
                    fraction[1] < border || fraction[1] > (1.-border) ||
                    fraction[2] < border || fraction[2] > (1.-border) ||
                    fraction[3] < border || fraction[3] > (1.-border)) {
                continue;
            }

            // Choose Max Variation
            gazeCalibrations[0] = max( abs(0.5-fraction[0]), gazeCalibrations[0] ); // get max from only left eye
            gazeCalibrations[1] = max( abs(0.5-fraction[1]), gazeCalibrations[1] );

            gazeCalibrations[0] = max( abs(0.5-fraction[2]), gazeCalibrations[0] ); // get max from either left or right
            gazeCalibrations[1] = max( abs(0.5-fraction[3]), gazeCalibrations[1] );

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

               // Shift centers relative to eye
               eye_centers[j].x += eyes[j].x + eyeROI.x;
               eye_centers[j].y += eyes[j].y + eyeROI.y;
        }

        // Draw Eye Center
        drawEyeCenter(eye_centers[0]);
        drawEyeCenter(eye_centers[1]);

        imshow("eye gaze",frame);
#endif

        if(gaze_count == (num_frames/3)) {
            circle(game_frame, gaze, 10, Scalar(0, 255, 124), 20);
            imshow( game_window, game_frame ); waitKey(1);
        }
        if(gaze_count == (num_frames*2/3)) {
            circle(game_frame, gaze, 10, Scalar(0, 255, 0), 20);
            imshow( game_window, game_frame ); waitKey(1);
        }

    } while(gaze_count < num_frames && waitKey(1)!='q');

    return;
}


