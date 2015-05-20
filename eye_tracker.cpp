//opencv
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
//c++
#include <stdio.h>
#include <iostream>
#include <sstream>

using namespace cv;
using namespace std;

const int DEFAULT_CAMERA=0;

//global variables
Mat frame; //current frame
int keyboard;

//function declarations
void processVideo();

int main(int argc, char* argv[])
{    
  //input data coming from a video
  processVideo();

  //destroy GUI windows
  destroyAllWindows();
  return EXIT_SUCCESS;
}

void processVideo() {
  //create the capture object
  VideoCapture capture(DEFAULT_CAMERA);
  if(!capture.isOpened()){
    //error in opening the camera
    cerr << "Unable to open camera: " << DEFAULT_CAMERA << endl;
    exit(EXIT_FAILURE);
  }

  //read input data. ESC or 'q' for quitting
  while( (char)keyboard != 'q' && (char)keyboard != 27 ){
    //read the current frame
    if(!capture.read(frame)) {
      cerr << "Unable to read next frame." << endl;
      cerr << "Exiting..." << endl;
      exit(EXIT_FAILURE);
    }

    //show the current frame and the fg masks
    imshow("Frame", frame);
    //get the input from the keyboard
    keyboard = waitKey( 30 );
  }
  //delete capture object
  capture.release();

}
