// ONE IMAGE

#include <iostream>
#include <cstdio>
#include <omp.h>
#include "opencv2/opencv.hpp"

int main(int argc, char** argv)
{   
  // read image
  cv::Mat image = cv::imread( "../testimg.png", cv::IMREAD_UNCHANGED );
  cv::Mat gray_image;
  cv::Mat blur;
  
  // display and wait for a key-press, then close the window
  cv::imshow( "image", image );
  int key = cv::waitKey( 0 );
  cv::destroyAllWindows();  

  double t0 = omp_get_wtime(); // start time
  
  //#pragma omp parallel for
  for ( int i = 0; i < image.rows; ++i ) {
    for ( int j = 0; j < image.cols; ++j ) {

      // get pixel at [i, j] as a <B,G,R> vector
      cv::Vec3b pixel = image.at<cv::Vec3b>( i, j );

      // extract the pixels as uchar (unsigned 8-bit) types (0..255)
      uchar b = pixel[0];
      uchar g = pixel[1];
      uchar r = pixel[2];   

      // Note: this is actually the slowest way to extract a pixel in OpenCV
      // Using pointers like this:
      //   uchar* ptr = (uchar*) image.data; // get raw pointer to the image data
      //   ...
      //   for (...) { 
      //       uchar* pixel = ptr + image.channels() * (i * image.cols + j);
      //       uchar b = *(pixel + 0); // Blue
      //       uchar g = *(pixel + 1); // Green
      //       uchar r = *(pixel + 2); // Red
      //       uchar a = *(pixel + 3); // (optional) if there is an Alpha channel
      //   }
      // is much faster
      
      uchar temp = r;
      r = b;
      b = temp;

      image.at<cv::Vec3b>( i, j ) = pixel;
      // or: 
      // image.at<cv::Vec3b>( i, j ) = {r, g, b};
    }
  }
  
  gray_image = cv::Mat::zeros( image.size(), CV_8U );
  //#pragma omp parallel for collapse(2) // <-- learn for yourself, what 'collapse' does
    for ( int i = 0; i < image.rows; ++i ) {
      for ( int j = 0; j < image.cols; ++j ) {

        // get pixel at [i, j] as a <B,G,R> vector
        cv::Vec3b pixel = image.at<cv::Vec3b>( i, j );

        // extract the pixels as uchar (unsigned 8-bit) types (0..255)
        uchar b = pixel[0];
        uchar g = pixel[1];
        uchar r = pixel[2];

        // convert to Grayscale
        for ( int k = 0; k < 150; k++ )
        {
          // some heavy workload here

          // write the pixel into the image at [i, j] as type 'unsigned 8-bit'
          gray_image.at<uchar>( i, j ) = 0.21 * r + 0.72 * g + 0.07 * b;
        }
      }
    }
    cv::GaussianBlur(gray_image, blur, cv::Size(15,15), 0);
    double t1 = omp_get_wtime();  // end time

  

  // display and wait for a key-press, then close the window
  cv::imshow( "image", gray_image );
  cv::waitKey(0);
  
  
  cv::imshow("blurred", blur);
  key = cv::waitKey( 0 );

  
  std::cout << "Processing took " << (t1 - t0) << " seconds" << std::endl;
  cv::destroyAllWindows();
} 