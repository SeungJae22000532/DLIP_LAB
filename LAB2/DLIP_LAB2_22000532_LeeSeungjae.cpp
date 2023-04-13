#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

Mat src;
int hmin = 0, hmax = 30, smin = 20, smax = 255, vmin = 0, vmax = 255; // set hsv min/max value to apply inRange function

int main() {
	VideoCapture cap("IR_DEMO_cut.avi"); // load a video
	VideoWriter videoWriter; // save a video

	if (!cap.isOpened()) { // if not success, exit the program 
		cout << "Cannot open the video cam\n";
		return -1;
	}

	float videoFPS = cap.get(cv::CAP_PROP_FPS);
	int videoWidth = cap.get(cv::CAP_PROP_FRAME_WIDTH);
	int videoHeight = cap.get(cv::CAP_PROP_FRAME_HEIGHT);

	videoWriter.open("DLIP_LAB2_22000532_LeeSeungjae.avi", cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), videoFPS, cv::Size(videoWidth, videoHeight), true);

	if (!videoWriter.isOpened())
	{
		std::cout << "Can't write video !!! check setting" << std::endl;
		return -1;
	}

	Mat image_disp, hsv, dst;
	vector<vector<Point>> contours;

	while (true) {
		bool bSuccess = cap.read(src); // read image of video per 10 milliseconds

		if (!bSuccess) { // if not success, break loop
			cout << "Cannot find a frame from video stream\n";
			break;
		}

		cvtColor(src, hsv, COLOR_BGR2HSV); // get hsv channer of origin image
		
		// apply pre-processing, gaussian blur as 5 x 5
		vector<Mat> pr(3);
		split(hsv, pr); // get h, s, and v channel respectively
		GaussianBlur(pr[2], pr[2], Size(5, 5), 0); // apply gaussian blur to only v channel
		merge(pr, hsv); // overwrite with blurred image

		cvtColor(hsv, hsv, COLOR_HSV2BGR);

		inRange(hsv, Scalar(MIN(hmin, hmax), MIN(smin, smax), MIN(vmin, vmax)), Scalar(MAX(hmin, hmax), MAX(smin, smax), MAX(vmin, vmax)), dst); /// set dst as the output of inRange

		// apply post-processing, closing of morphology
		Mat element = getStructuringElement(MORPH_RECT, Size(5, 5));
		morphologyEx(dst, dst, CV_MOP_CLOSE, element); // apply closing to eliminate internal noise
		dilate(dst, dst, element); // dilate contour

		findContours(dst, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE); // find all contour

		if (contours.size() > 0) {
			// find the contour with the largest area
			double maxArea = 0;
			int maxArea_idx = 0;

			for (int i = 0; i < contours.size(); i++) {
				if (contourArea(contours[i]) > maxArea) {
					maxArea = contourArea(contours[i]);
					maxArea_idx = i;
				}
			}

			src.copyTo(image_disp);

			if (maxArea > 3000) {
				// draw the largest contour on original image
				drawContours(image_disp, contours, maxArea_idx, Scalar(255, 255, 255), 2, 8);
				// draw the contour box on original image
				Rect boxPoint = boundingRect(contours[maxArea_idx]);
				rectangle(image_disp, boxPoint, Scalar(255, 0, 255), 3);

				vector<Point> internal; // this variable stores the internal points of the largest contour

				for (int y = boxPoint.y; y < boxPoint.y + boxPoint.height; y++) { // access y-axis in the contour box
					for (int x = boxPoint.x; x < boxPoint.x + boxPoint.width; x++) { // access x-axis in the contour box
						if (dst.at<uchar>(Point(x, y)) > 0) // if that point is activatied, then it is the internal point
							internal.push_back(Point(x, y));
					}
				}

				vector<float> tem(internal.size());
				float max_tem = 0.0, avg_tem = 0.0;

				for (int i = 0; i < internal.size(); i++) {
					tem[i] = int(pr[2].at<uchar>(internal[i])) * 1.0; // find the value of each internal point
					tem[i] = (tem[i] / 255.0) * 15.0 + 25.0; // convert temperature value between 0 - 255 to 25 - 40
				}
				
				sort(tem.rbegin(), tem.rend()); // sort in descending order
				max_tem = tem[0]; // the biggest temperature is the first value of tem vector
				for (int i = 0; i < tem.size() * 0.01; i++) { // use the data within the Top 2% of the temperature in descending order
					avg_tem += tem[i];
				}
				avg_tem /= (tem.size() * 0.01); // estimate the average temperature

				if(avg_tem <= 38.0)
					putText(image_disp, format("Max: %.0f    Avg: %.0f", max_tem, avg_tem), Point(50, 50), FONT_HERSHEY_COMPLEX, 1, Scalar(255, 255, 255), 1, 8);
				else { // if the average temperature is above 38.0, then show WARNING message
					putText(image_disp, format("Max: %.0f    Avg: %.0f", max_tem, avg_tem), Point(50, 50), FONT_HERSHEY_COMPLEX, 1, Scalar(0, 0, 255), 1, 8);
					putText(image_disp, "WARNING", Point(50, 120), FONT_HERSHEY_COMPLEX, 2, Scalar(0, 0, 255), 1, 8);
				}
			}

			namedWindow("Temperature", 0);
			imshow("Temperature", image_disp);
		}

		videoWriter << image_disp;

		char c = (char)waitKey(10);
		if (c == 27)
			break;
	}



	return 0;
}
