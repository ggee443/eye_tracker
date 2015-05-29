#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

//#include <mgl2/mgl.h>

#include <iostream>
#include <queue>
#include <stdio.h>

#include "constants.h"
#include "helpers.h"


using namespace cv;

Point p_pred;

// Pre-declarations
Mat floodKillEdges(Mat &mat);
Point kalmanPostProcess(Point&);

#pragma mark Visualization

/*
template<typename T> mglData *matToData(const Mat &mat) {
  mglData *data = new mglData(mat.cols,mat.rows);
  for (int y = 0; y < mat.rows; ++y) {
    const T *Mr = mat.ptr<T>(y);
    for (int x = 0; x < mat.cols; ++x) {
      data->Put(((mreal)Mr[x]),x,y);
    }
  }
  return data;
}

void plotVecField(const Mat &gradientX, const Mat &gradientY, const Mat &img) {
  mglData *xData = matToData<double>(gradientX);
  mglData *yData = matToData<double>(gradientY);
  mglData *imgData = matToData<float>(img);
  
  mglGraph gr(0,gradientX.cols * 20, gradientY.rows * 20);
  gr.Vect(*xData, *yData);
  gr.Mesh(*imgData);
  gr.WriteFrame("vecField.png");
  
  delete xData;
  delete yData;
  delete imgData;
}*/

#pragma mark Helpers

Point unscalePoint(Point p, Rect origSize) {
  float ratio = (((float)kFastEyeWidth)/origSize.width);
  int x = round(p.x / ratio);
  int y = round(p.y / ratio);
  return Point(x,y);
}

void scaleToFastSize(const Mat &src,Mat &dst) {
  resize(src, dst, Size(kFastEyeWidth,(((float)kFastEyeWidth)/src.cols) * src.rows));
}

Mat computeMatXGradient(const Mat &mat) {
  Mat out(mat.rows,mat.cols,CV_64F);
  
  for (int y = 0; y < mat.rows; ++y) {
    const uchar *Mr = mat.ptr<uchar>(y);
    double *Or = out.ptr<double>(y);
    
    Or[0] = Mr[1] - Mr[0];
    for (int x = 1; x < mat.cols - 1; ++x) {
      Or[x] = (Mr[x+1] - Mr[x-1])/2.0;
    }
    Or[mat.cols-1] = Mr[mat.cols-1] - Mr[mat.cols-2];
  }
  
  return out;
}

#pragma mark Main Algorithm

//--(!) Strip this to bare essentials --> find out what is most expensive, then make it less expensive


void testPossibleCentersFormula(int x, int y, const Mat &weight,double gx, double gy, Mat &out) {
  // for all possible centers
  for (int cy = 0; cy < out.rows; ++cy) {
    double *Or = out.ptr<double>(cy);
    const unsigned char *Wr = weight.ptr<unsigned char>(cy);
    for (int cx = 0; cx < out.cols; ++cx) {
      if (x == cx && y == cy) {
        continue;
      }
      // create a vector from the possible center to the gradient origin
      double dx = x - cx;
      double dy = y - cy;
      // normalize d
      double magnitude = sqrt((dx * dx) + (dy * dy));
      dx = dx / magnitude;
      dy = dy / magnitude;
      double dotProduct = dx*gx + dy*gy;
      dotProduct = std::max(0.0,dotProduct);
      // square and multiply by the weight
      if (kEnableWeight) {
        Or[cx] += dotProduct * dotProduct * (Wr[cx]/kWeightDivisor);
      } else {
        Or[cx] += dotProduct * dotProduct;
      }
    }
  }
}

Point findEyeCenter(Mat &face, Rect eye, std::string debugWindow) {
  Mat eyeROIUnscaled = face(eye);
  Mat eyeROI;
  scaleToFastSize(eyeROIUnscaled, eyeROI);
  // draw eye region
//  rectangle(face,eye,1234);
  //-- Find the gradient
  Mat gradientX = computeMatXGradient(eyeROI);
  Mat gradientY = computeMatXGradient(eyeROI.t()).t();
  //-- Normalize and threshold the gradient
  // compute all the magnitudes
  Mat mags = matrixMagnitude(gradientX, gradientY);
  //compute the threshold
  double gradientThresh = computeDynamicThreshold(mags, kGradientThreshold);
  //double gradientThresh = kGradientThreshold;
  //double gradientThresh = 0;
  //normalize
  for (int y = 0; y < eyeROI.rows; ++y) {
    double *Xr = gradientX.ptr<double>(y), *Yr = gradientY.ptr<double>(y); // not vector?
    const double *Mr = mags.ptr<double>(y);
    for (int x = 0; x < eyeROI.cols; ++x) {
      double gX = Xr[x], gY = Yr[x];
      double magnitude = Mr[x];
      if (magnitude > gradientThresh) {
        Xr[x] = gX/magnitude;
        Yr[x] = gY/magnitude;
      } else {
        Xr[x] = 0.0;
        Yr[x] = 0.0;
      }
    }
  }
//  imshow(debugWindow,gradientX);
  //-- Create a blurred and inverted image for weighting
  Mat weight;
  GaussianBlur( eyeROI, weight, Size( kWeightBlurSize, kWeightBlurSize ), 0, 0 );
  for (int y = 0; y < weight.rows; ++y) {
    unsigned char *row = weight.ptr<unsigned char>(y);
    for (int x = 0; x < weight.cols; ++x) {
      row[x] = (255 - row[x]);
    }
  }
  //imshow(debugWindow,weight);
  //-- Run the algorithm!
  Mat outSum = Mat::zeros(eyeROI.rows,eyeROI.cols,CV_64F);
  // for each possible gradient location
  // Note: these loops are reversed from the way the paper does them
  // it evaluates every possible center for each gradient location instead of
  // every possible gradient location for every center.
//  printf("Eye Size: %ix%i\n",outSum.cols,outSum.rows);
  for (int y = 0; y < weight.rows; ++y) {
    const double *Xr = gradientX.ptr<double>(y), *Yr = gradientY.ptr<double>(y);
    for (int x = 0; x < weight.cols; ++x) {
      double gX = Xr[x], gY = Yr[x];
      if (gX == 0.0 && gY == 0.0) {
        continue;
      }
      testPossibleCentersFormula(x, y, weight, gX, gY, outSum);
    }
  }
  // scale all the values down, basically averaging them
  double numGradients = (weight.rows*weight.cols);
  Mat out;
  outSum.convertTo(out, CV_32F,1.0/numGradients);
  //imshow(debugWindow,out);
  //-- Find the maximum point
  Point maxP;
  double maxVal;
  minMaxLoc(out, NULL,&maxVal,NULL,&maxP);
  //-- Flood fill the edges
  if(kEnablePostProcess) {
    Mat floodClone;
    //double floodThresh = computeDynamicThreshold(out, 1.5);
    double floodThresh = maxVal * kPostProcessThreshold;
    threshold(out, floodClone, floodThresh, 0.0f, THRESH_TOZERO);
    if(kPlotVectorField) {
      //plotVecField(gradientX, gradientY, floodClone);
      imwrite("eyeFrame.png",eyeROIUnscaled);
    }
    Mat mask = floodKillEdges(floodClone);
    //imshow(debugWindow + " Mask",mask);
    //imshow(debugWindow,out);
    // redo max
    minMaxLoc(out, NULL,&maxVal,NULL,&maxP,mask);
  }
  return unscalePoint(maxP,eye);
}

#pragma mark Postprocessing

bool floodShouldPushPoint(const Point &np, const Mat &mat) {
  return inMat(np, mat.rows, mat.cols);
}

// returns a mask
Mat floodKillEdges(Mat &mat) {
  rectangle(mat,Rect(0,0,mat.cols,mat.rows),255);
  
  Mat mask(mat.rows, mat.cols, CV_8U, 255);
  std::queue<Point> toDo;
  toDo.push(Point(0,0));
  while (!toDo.empty()) {
    Point p = toDo.front();
    toDo.pop();
    if (mat.at<float>(p) == 0.0f) {
      continue;
    }
    // add in every direction
    Point np(p.x + 1, p.y); // right
    if (floodShouldPushPoint(np, mat)) toDo.push(np);
    np.x = p.x - 1; np.y = p.y; // left
    if (floodShouldPushPoint(np, mat)) toDo.push(np);
    np.x = p.x; np.y = p.y + 1; // down
    if (floodShouldPushPoint(np, mat)) toDo.push(np);
    np.x = p.x; np.y = p.y - 1; // up
    if (floodShouldPushPoint(np, mat)) toDo.push(np);
    // kill it
    mat.at<float>(p) = 0.0f;
    mask.at<uchar>(p) = 0;
  }
  return mask;
}

// pass point through kalman filter
Point kalmanPostProcess(Point &pp)
{


    return p_pred;
}


