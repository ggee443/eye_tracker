#include <opencv2/highgui/highgui.hpp>
#include <stdio.h>

using namespace std;
using namespace cv;


int main( void )
{
  VideoCapture capture(0);  //sudo apt-get install v4l2ucp v4l-utils libv4l-dev


  Mat mat;
  Scalar stdMagnGrad, meanMagnGrad;
  meanStdDev(mat, meanMagnGrad, stdMagnGrad);




/*
  Mat frame;
  int c;

  //-- 2. Read the video stream
  if( capture.isOpened() )
  {      
    while((char)c != 'c')
    {
      //-- 3. Apply the classifier to the frame
      if( capture.read(frame) ) {
            flip(frame, frame, 1);
            imshow("Frame", frame);
        }
      else
      { printf(" --(!) No captured frame -- Break!\n"); break; }

      c = waitKey(1);

    }
  }
  */


  return 0;
}

// g++ toy.cpp -o toy -L/usr/local/lib/ -lopencv_core -lopencv_highgui
