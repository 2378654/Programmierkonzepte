// from camera

#include <iostream>
#include <cstdio>
//#include <omp.h>
#include <string>

#include "opencv2/opencv.hpp"

int main(int argc, char** argv)
{   
  // initialize the camera
  cv::VideoCapture camera = cv::VideoCapture( 1 ); 
  // "1" here is the device-index. 
  // I have several cameras on my PC, "1" is my webcam

  // change camera-capture properties, in this case, the frame size, because OpenCV uses 640x480 by default.
  camera.set( cv::CAP_PROP_FRAME_HEIGHT, 720 );
  camera.set( cv::CAP_PROP_FRAME_WIDTH, 1280 );
  
  // prepare empty images
  cv::Mat image;
  cv::Mat gray_image;

  // start an endless loop for the camera
  while ( true ) {

    // capture image from camera
    camera >> image;

    // check if the image is not empty
    if ( image.empty() ) {
      break;
    }

    //double t0 = omp_get_wtime();  // start time

    // init gray image (empty, of same size as `image`, one channel, 8-bit):
    gray_image = cv::Mat::zeros( image.size(), CV_8U );

    #pragma omp parallel for collapse(2) // <-- learn for yourself, what 'collapse' does
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

    //double t1 = omp_get_wtime();  // end time

    // calculate FPS 
    //double fps = 1 / (t1 - t0);

    // show fps in scren corner, using the put-text function
    //std::string fps_text = "FPS: " + std::to_string( static_cast<int>(fps) );
    //cv::putText( gray_image, fps_text, { 5, 40 }, cv::FONT_HERSHEY_DUPLEX, 1, { 255,255,255 }, 2 );

    // display the image
    cv::imshow( "image", gray_image );

    // wait 1ms for a key-press
    int key = cv::waitKey( 1 );

    if ( key == 27 ) // ESC key
    {
      break; // exit loop
    }
  }

  // close all windows
  cv::destroyAllWindows();
}