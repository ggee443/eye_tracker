#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <iostream>
#include <queue>
#include <stdio.h>

#include "constants.h"

using namespace std;
using namespace cv;

bool rectInImage(Rect rect, Mat* image) {
  return rect.x > 0 && rect.y > 0 && rect.x+rect.width < image->cols &&
  rect.y+rect.height < image->rows;
}

bool inMat(Point p,int rows,int cols) {
  return p.x >= 0 && p.x < cols && p.y >= 0 && p.y < rows;
}

Mat matrixMagnitude(const Mat &matX, const Mat &matY) {
  Mat mags(matX.rows,matX.cols,CV_64F);
  for (int y = 0; y < matX.rows; ++y) {
    const double *Xr = matX.ptr<double>(y), *Yr = matY.ptr<double>(y);
    double *Mr = mags.ptr<double>(y);
    for (int x = 0; x < matX.cols; ++x) {
      double gX = Xr[x], gY = Yr[x];
      double magnitude = sqrt((gX * gX) + (gY * gY));
      Mr[x] = magnitude;
    }
  }
  return mags;
}

double computeDynamicThreshold(const Mat &mat, double stdDevFactor) {
  Scalar stdMagnGrad, meanMagnGrad;
  meanStdDev(mat, meanMagnGrad, stdMagnGrad);
  double stdDev = stdMagnGrad[0] / sqrt(mat.rows*mat.cols);
  return stdDevFactor * stdDev + meanMagnGrad[0];
}

void waitForSpace() {
    int cc;
    while((char)cc != ' ') {
        cc = waitKey(1);
    }
    cc = 0;
}


