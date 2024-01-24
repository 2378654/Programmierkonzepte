#include <iostream>
#include <cstdio>
#include <omp.h>
#include "opencv2/opencv.hpp"
#include <mpi.h>


int main(int argc, char** argv)
{   
  MPI_Init(&argc,&argv);
  char processor_name[MPI_MAX_PROCESSOR_NAME];
  int name_len;
  MPI_Get_processor_name(processor_name, &name_len);

  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  // Get the rank of the process
  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  // read image
  cv::Mat rgb = cv::imread( "./960x540.jpg", cv::IMREAD_UNCHANGED ); 
  /*
  Getestete Bilder
  960x540.jpg
  1920x1080.jpg
  3840x2160.jpg
  */

  cv::Mat gray_image;
  
  int key = cv::waitKey( 0 ); 

  double start_comp = omp_get_wtime();
  double start_rgb = omp_get_wtime(); //Startzeit für RGB Bild in Variable schreiben
  
  #pragma omp parallel for
  for ( int i = 0; i < rgb.rows; ++i ) {
    for ( int j = 0; j < rgb.cols; ++j ) {

      cv::Vec3b pixel = rgb.at<cv::Vec3b>( i, j );

      uchar b = pixel[0];
      uchar g = pixel[1];
      uchar r = pixel[2];   
      uchar temp = r;
      r = b;
      b = temp;

      rgb.at<cv::Vec3b>( i, j ) = pixel;
    }
  }
  double ende_rgb = omp_get_wtime();  //Endzeit für RGB Bild in Variable schreiben
  gray_image = cv::Mat::zeros( rgb.size(), CV_8U );
  
  double start_gray = omp_get_wtime(); //Startzeit für Grayscale Bild in Variable schreiben

  #pragma omp parallel for
    for ( int i = 0; i < rgb.rows; ++i ) {
      for ( int j = 0; j < rgb.cols; ++j ) {

        cv::Vec3b pixel = rgb.at<cv::Vec3b>( i, j );

        uchar b = pixel[0];
        uchar g = pixel[1];
        uchar r = pixel[2];
        
        // convert to Grayscale
        for ( int k = 0; k < 150; k++ )
        {
          gray_image.at<uchar>( i, j ) = 0.21 * r + 0.72 * g + 0.07 * b;
        }
      }
    }
double ende_gray = omp_get_wtime();   //Endzeit für Grayscale Bild in Variable schreiben

cv::Mat blur = gray_image.clone();  // Initialisiere blur mit einer Kopie von gray_image

double start_blur = omp_get_wtime(); //Startzeit für Blured Bild in Variable schreiben
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
double ende_blur = omp_get_wtime();  //Endzeit für Blured Bild in Variable schreiben
double ende_comp = omp_get_wtime();  //Endzeit für Gesamte Ausführung in Variable schreiben

  //cv::GaussianBlur(gray_image, blur, cv::Size(15,15), 0); --> auch mit eigener OpenCV Funktion möglich
    
  cv::imshow( "image", rgb ); 
  cv::waitKey(0);
  cv::imwrite( "RGB_Image.jpg", rgb );
  
  cv::imshow( "Grayscale", gray_image );
  cv::waitKey(0);
  cv::imwrite( "Grayscale_Image.jpg", gray_image );  

  cv::imshow("Blurred", blur);
  cv::waitKey(0);
  cv::imwrite( "Blurred_Image.jpg", blur );


  printf("=====Processor: %s, Rank: %d out of %d Processors=====\n", processor_name, world_rank, world_size);
  //std::cout << "==========================================\n";
  std::cout << "RGB: "<< (ende_rgb - start_rgb) << "\n"; //Laufzeit RGB
  std::cout << "Gray: " << (ende_gray - start_gray) << "\n"; //Laufzeit Grayscale
  std::cout << "Blur: "<< (ende_blur - start_blur) << "\n"; //Laufzeit Blured Bild
  std::cout << "Gesamt: "<< (ende_comp - start_comp) << "\n";
  std::cout << "Threads Used: " << omp_get_max_threads() << "\n";
  
  cv::destroyAllWindows();
  MPI_Finalize();
}