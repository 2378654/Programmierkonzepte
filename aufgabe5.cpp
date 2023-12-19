#include <iostream>
#include <cstdio>
#include <omp.h>
#include "opencv2/opencv.hpp"

int main(int argc, char** argv)
{   
  // read image
  cv::Mat rgb = cv::imread( "./1920x1080.jpg", cv::IMREAD_UNCHANGED ); 
  /*
  Getestete Bilder
  960x540.jpg
  1920x1080.jpg
  3840x2160.jpg
  */

  cv::Mat grey_image;
  
  int key = cv::waitKey( 0 ); 

  double start_rgb = omp_get_wtime();
  
  #pragma omp parallel for
  for ( int i = 0; i < rgb.rows; ++i ) {
    for ( int j = 0; j < rgb.cols; ++j ) {

      // get pixel at [i, j] as a <B,G,R> vector
      cv::Vec3b pixel = rgb.at<cv::Vec3b>( i, j );

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

      rgb.at<cv::Vec3b>( i, j ) = pixel;
    }
  }
  double ende_rgb = omp_get_wtime();  // end time
  grey_image = cv::Mat::zeros( rgb.size(), CV_8U );
  
  double start_gray = omp_get_wtime();

  #pragma omp parallel for
    for ( int i = 0; i < rgb.rows; ++i ) {
      for ( int j = 0; j < rgb.cols; ++j ) {

        // get pixel at [i, j] as a <B,G,R> vector
        cv::Vec3b pixel = rgb.at<cv::Vec3b>( i, j );

        // extract the pixels as uchar (unsigned 8-bit) types (0..255)
        uchar b = pixel[0];
        uchar g = pixel[1];
        uchar r = pixel[2];

        // convert to Grayscale
        for ( int k = 0; k < 150; k++ )
        {
          // some heavy workload here

          // write the pixel into the image at [i, j] as type 'unsigned 8-bit'
          grey_image.at<uchar>( i, j ) = 0.21 * r + 0.72 * g + 0.07 * b;
        }
      }
    }
double ende_gray = omp_get_wtime();  // end time

cv::Mat blur = grey_image.clone();  // Initialisiere blur mit einer Kopie von grey_image

double start_blur = omp_get_wtime();
#pragma omp parallel for
for (int x = 1; x < blur.rows - 1; ++x) {
    for (int y = 1; y < blur.cols - 1; ++y) {
        float sum = blur.at<uchar>(x - 1, y + 1) + // Top left
                    blur.at<uchar>(x + 0, y + 1) + // Top center
                    blur.at<uchar>(x + 1, y + 1) + // Top right
                    blur.at<uchar>(x - 1, y + 0) + // Mid left
                    blur.at<uchar>(x + 0, y + 0) + // Current pixel
                    blur.at<uchar>(x + 1, y + 0) + // Mid right
                    blur.at<uchar>(x - 1, y - 1) + // Low left
                    blur.at<uchar>(x + 0, y - 1) + // Low center
                    blur.at<uchar>(x + 1, y - 1);  // Low right

        blur.at<uchar>(x, y) = sum / 9;
    }
}
double ende_blur = omp_get_wtime();  // end time
    
    cv::Mat floatImage;
    blur.convertTo(floatImage, CV_32F);

    // Konvertiere das Ergebnis zurück in den 8-Bit-Bereich für die Anzeige
    cv::Mat outputImage;
    blur.convertTo(outputImage, CV_8U);
    


    //cv::GaussianBlur(grey_image, blur, cv::Size(15,15), 0);
    
  

  cv::imshow( "image", rgb ); 
  cv::waitKey(0);
  
  cv::imshow( "Greyscale", grey_image );
  cv::waitKey(0);
  
  
  cv::imshow("Blurred", blur);
  cv::waitKey(0);

  
  std::cout << "RGB: "<< (ende_rgb - start_rgb) << "\n";
  std::cout << "Gray: " << (ende_gray - start_gray) << "\n";
  std::cout << "Blur: "<< (ende_blur - start_blur) << "\n";
  cv::destroyAllWindows();
}