#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

// declare global variables
Mat src, src_gray, src_blur, src_thres, src_morph, dst;
Mat open_element, close_element, element;
Mat draw, draw_gray;

vector<vector<Point>> contours;
vector<Vec4i> hierarchy;
vector<int> nnb(5);

// declare user defined functions
void pre_filter(int, void*);
void pre_threshold(int, void*);
void pre_morphology(int, void*);
void find_contours(int, void*);
void draw_contours(vector<vector<Point> > contours_poly, vector<Rect> boundRect, vector<int> children);

int main() {
	src = imread("Lab_GrayScale_TestImage.jpg"); // open a image as BGR channer
	cvtColor(src, src_gray, COLOR_BGR2GRAY); // convert the BGR image to gray image

	pre_filter(0, 0); // 1) apply appropriate filters to enhance image
	pre_threshold(0, 0); // 2) apply appropriate threshold
	pre_morphology(0, 0); // 3) apply the appropriate morphology method to segment parts
	find_contours(0, 0); // 4) find contours

	draw = Mat::zeros(src.size(), CV_8UC3); // define a drawing board as BGR channer and all values are zero

	// declare contour additional elements
	vector<vector<Point>> contours_poly(contours.size()); // variable that store approximate points of segmented objects
	vector<Rect> boundRect(contours.size()); // variable that store minimum bounding boxes for segmented objects
	vector<int> children(contours.size()); // variable that store the number of internal boundaries

	draw_contours(contours_poly, boundRect, children); // 5) draw segmented objects
	cvtColor(draw, draw_gray, COLOR_BGR2GRAY); // convert the BGR drawing board to gray image
	dst = src.clone(); // deep copy of the image
	draw.copyTo(dst, draw_gray); // draw detected bounding boxes in the image

	// count the number of each parts
	cout << "M5 Bolt\t\t= " << nnb[1] << "\n";
	cout << "M6 Bolt\t\t= " << nnb[0] << "\n";
	cout << "M5 Hex Nut\t= " << nnb[4] << "\n";
	cout << "M6 Hex Nut\t= " << nnb[2] << "\n";
	cout << "M5 Rect Nut\t= " << nnb[3] << "\n";
	
	//namedWindow("src", WINDOW_FREERATIO);
	//imshow("src", src);
	//namedWindow("src_gray", WINDOW_FREERATIO);
	//imshow("src_gray", src_gray);
	//namedWindow("src_blur", WINDOW_FREERATIO);
	//imshow("src_blur", src_blur);
	//namedWindow("src_threshold", WINDOW_FREERATIO);
	//imshow("src_threshold", src_thres);
	//namedWindow("src_morphology", WINDOW_FREERATIO);
	//imshow("src_morphology", src_morph);
	//namedWindow("draw", WINDOW_FREERATIO);
	//imshow("draw", draw);
	//namedWindow("draw_gray", WINDOW_FREERATIO);
	//imshow("draw_gray", draw_gray);
	namedWindow("dst", WINDOW_FREERATIO);
	imshow("dst", dst);
	waitKey(0);
	
	return 0;
}

void pre_filter(int, void*) {
	GaussianBlur(src_gray, src_blur, Size(5, 5), 0); // blur to the gray image
}

void pre_threshold(int, void*) {
	threshold(src_blur, src_thres, 118, 255, THRESH_BINARY); // apply threshold to the gray image
}

void pre_morphology(int, void*) {
	open_element = getStructuringElement(MORPH_RECT, Size(6, 6));
	close_element = getStructuringElement(MORPH_RECT, Size(4, 4));
	element = getStructuringElement(MORPH_RECT, Size(13, 13));

	morphologyEx(src_thres, src_morph, CV_MOP_OPEN, open_element); // apply opening to eliminate external noise
	morphologyEx(src_morph, src_morph, CV_MOP_CLOSE, close_element); // apply closing to eliminate internal noise
	dilate(src_morph, src_morph, element); // dilate to connect the broken part
}

void find_contours(int, void*) {
	findContours(src_morph, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE); // find contours
}

void draw_contours(vector<vector<Point> > contours_poly, vector<Rect> boundRect, vector<int> children) {
	for (int i = 0; i < contours.size(); i++) {
		if (hierarchy[i][3] != -1) children[hierarchy[i][3]]++; // find how many internal boundaries each contour
	}

	for (int i = 0; i < contours.size(); i++) {
		approxPolyDP(contours[i], contours_poly[i], 3, true); // approximate segmented object using approxPolyDP function
		boundRect[i] = boundingRect(contours_poly[i]); // find bounding boxes for each segmented object using boundingRect function
	}

	for (int i = 0; i < contours.size(); i++) {
		int area = contourArea(contours_poly[i]); // find the area of the approximated contour using contourArea function

		if (children[i] == 0) {
			if (area > 11500) {
				nnb[0]++; // the number of M6 bolts
				rectangle(draw, boundRect[i].tl(), boundRect[i].br(), Scalar(255, 0, 0), 2);
			}
			else if (area > 9000) {
				nnb[1]++; // the number of M5 bolts
				rectangle(draw, boundRect[i].tl(), boundRect[i].br(), Scalar(0, 255, 0), 2);
			}
		}
		else {
			if (children[i] > 1) area /= children[i]; // if the objects are overlapped, divide the area by the number of overlapped objects
			if (area > 7000) {
				// the number of M6 hex nuts
				if (children[i] > 1) nnb[2] += children[i];
				else nnb[2]++;
				rectangle(draw, boundRect[i].tl(), boundRect[i].br(), Scalar(0, 0, 255), 2);
				for (int idx = hierarchy[i][2]; idx >= 0; idx = hierarchy[idx][0]) {
					rectangle(draw, boundRect[idx].tl(), boundRect[idx].br(), Scalar(0, 0, 255), 2);
				}
			}
			else if (area > 5300) {
				// the number of M5 rect nuts
				if (children[i] > 1) nnb[3] += children[i];
				else nnb[3]++;
				rectangle(draw, boundRect[i].tl(), boundRect[i].br(), Scalar(255, 255, 0), 2);
				for (int idx = hierarchy[i][2]; idx >= 0; idx = hierarchy[idx][0]) {
					rectangle(draw, boundRect[idx].tl(), boundRect[idx].br(), Scalar(255, 255, 0), 2);
				}
			}
			else if (area > 4000) {
				// the number of M5 hex nuts
				if (children[i] > 1) nnb[4] += children[i];
				else nnb[4]++;
				rectangle(draw, boundRect[i].tl(), boundRect[i].br(), Scalar(255, 0, 255), 2);
				for (int idx = hierarchy[i][2]; idx >= 0; idx = hierarchy[idx][0]) {
					rectangle(draw, boundRect[idx].tl(), boundRect[idx].br(), Scalar(255, 0, 255), 2);
				}
			}
		}
	}
}