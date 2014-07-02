#include "stdafx.h"
#include "opencv2\core\core.hpp"
#include "opencv2\highgui\highgui.hpp"
#include "opencv2\imgproc\imgproc.hpp"
#include <iostream>
#include <vector>

using namespace cv;
using namespace std;

/// Global Variables
Mat img; Mat cutImage; Mat templ; Mat result; Mat gray;
RNG rng(12345);

const int alpha_slider_max = 99;
int alpha_slider=0;
int choice = 0;

/// Function Headers
Vec3b getPixelColorAt(Mat* img, int x, int y);
float getPixelColorAtf(Mat* img, int x, int y);
void setPixelColorAt(Mat* img, int x, int y, uchar R, uchar G, uchar B);
void setPixelColorAt(Mat* img, int x, int y, float color);
float getAvgPixelColor(Mat* img, int x, int y, int step, int range);
float getAverageImageColor(Mat* img);
void normalizeImage(Mat* img, float rate, int iterations);
void polarizeImage(Mat* img);
int getMaxColor(Mat* img);
int getMinColor(Mat* img);
void binarizeAndNormalize(Mat* img, float tolerance);
void binarizeAndNormalize(Mat* img, float tolerance, Vec2i step);
void thresholdArea(Mat* img, Rect area, int min, int max, bool inverted);
int getMinColorInArea(Mat* img, Rect area);
int getMaxColorInArea(Mat* img, Rect area);
void getGridAnswers(string windowName);
void GetImageFromPoints();
struct line2 { Vec2f s, e; };
double CCW(Vec2f a, Vec2f b, Vec2f c);
int middle(int a, int b, int c);
int intersect(line2 a, line2 b);
void CallBackFunc(int event, int x, int y, int flags, void* userdata);
void findGrid(string windowName, bool hasBounds);
float distance(float p1x, float p1y, float p2x, float p2y);
bool contains(Point test, vector<Point> points) ;
Vec4i getWeightedBounds(vector<Point> points, float toleranceX, float toleranceY);
void on_trackbar( int, void* );
float getLineHorizontalRatio(Point A, Point B);
float getLineVerticalRatio(Point A, Point B);

//quelques variables en rapport avec le découpage de l'image.
vector<Point2f> storedPoints;
Vec4i mouseBounds = Vec4i(100, 100, 200, 200);

/** @function main */
int _tmain(int argc, _TCHAR* argv[])
{	

	cout<<"0 - traitement complet"<<endl;
	cout<<"1 - traitement avec encadrement souris (2 points nécessaires)"<<endl;
	cout<<"2 - traitement sur image déjà découpée"<<endl;
	cout<<"3 - traitement sur sans detection de note (avec slider)"<<endl;
	cin>>choice;
	if(choice==0)
	{
		string imgName;
		cout<<"Tappez le nom de l'image à analyser:"<<endl;
		cin>>imgName;
		img = imread( "test/"+imgName);

		if(!img.data)
		{
			img = imread( "test/001.jpg");
			if(!img.data)
			{
				return 0;
			}
		}

		findGrid("Lines", false);
		GetImageFromPoints();
		waitKey(0);
		return 0;
	}
	else if(choice==1)
	{
		/// On charge l'image à analyser.
		string imgName;
		cout<<"Tappez le nom de l'image à analyser:"<<endl;
		cin>>imgName;
		img = imread( "test/"+imgName);

		if(!img.data)
		{
			img = imread( "test/001.jpg");
			if(!img.data)
			{
				return 0;
			}
		}

		namedWindow("original", 1);
		setMouseCallback("original", CallBackFunc, NULL);

		waitKey(0);
		return 0;
	}
	else
	{
		/// Create Windows
		namedWindow(":D", 1);
		createTrackbar( "Image", ":D", &alpha_slider, alpha_slider_max, on_trackbar );
		/// Show some stuff
		on_trackbar( alpha_slider, 0 );
		waitKey(0);
		return 0;
	}
}

//gestion de la trackBar
void on_trackbar( int, void* )
{
	if(choice==3)
	{
		string imgName = "test/0";
		if(alpha_slider<10)
			imgName="test/00";

		imgName += std::to_string(alpha_slider);
		imgName += ".jpg";
		img = imread(imgName);

		cout<<"Image:"<<imgName<<endl;
		cout<<"alpha_slider: "<<alpha_slider<<endl;

		if( img.data ) { 
			findGrid(":D", false);
			//waitKey(0);
		}
		else
		{
			cout<<"Image not found: "<<imgName<<endl;
		}
	}
	else
	{
		string imgName = "test/corrected0";
		if(alpha_slider<10)
			imgName="test/corrected00";

		imgName += std::to_string(alpha_slider);
		imgName += ".jpg";
		img = imread(imgName);

		cout<<"Image:"<<imgName<<endl;
		cout<<"alpha_slider: "<<alpha_slider<<endl;

		if( img.data ) { 
			cutImage = img.clone();
			getGridAnswers(":D");
		}
		else
		{
			cout<<"Image not found: "<<imgName<<endl;
		}
	}
}

void findGrid(string windowName, bool hasBounds)
{

	Mat imgClone, dst, color_dst;
	imgClone = img.clone();
	Canny( imgClone, dst, 50, 200, 3 );
	cvtColor( dst, color_dst, COLOR_GRAY2BGR );

	int maxLineGap = 20;
	vector<Point2f> bestResult;
	int lowestScore = 99999999999;

	while(true)
	{
		bool isHorizontal = false;
		if(img.size().width>img.size().height)
		isHorizontal = true;

		vector<Vec4i> lines;
		HoughLinesP( dst, lines, 1, CV_PI/180, 80, 30, maxLineGap );

		//cout<<"Found "<<lines.size()<<" lines..."<<endl;

		vector<Point> points;
		map<float, Vec4i> linesByDistance;
		for( size_t i = 0; i < lines.size(); i++ )
		{
			int color = 0;
			  // line( color_dst, Point(lines[i][0], lines[i][1]),
			 //      Point(lines[i][2], lines[i][3]), Scalar(255-color, 0, color), 1, 8 );
			//circle(color_dst, Point(lines[i][0], lines[i][1]), 3, Scalar(255-color, 0, color), 1);
			//circle(color_dst, Point(lines[i][2], lines[i][3]), 3, Scalar(255-color, 0, color), 1);
			float dist =distance(lines[i].val[0], lines[i].val[1], lines[i].val[2], lines[i].val[3]);
			linesByDistance[dist] = lines[i];

			if(!isHorizontal)
			{
				if(distance(lines[i][0], lines[i][1], lines[i][2], lines[i][3])>150 && getLineVerticalRatio(Point(lines[i][0], lines[i][1]), Point( lines[i][2], lines[i][3]))>2)
				{
					points.push_back(Point(lines[i][0], lines[i][1]));
					points.push_back(Point(lines[i][2], lines[i][3]));
				}
			}
			else
			{
				if(distance(lines[i][0], lines[i][1], lines[i][2], lines[i][3])>150 && getLineHorizontalRatio(Point(lines[i][0], lines[i][1]), Point( lines[i][2], lines[i][3]))>2)
				{
					points.push_back(Point(lines[i][0], lines[i][1]));
					points.push_back(Point(lines[i][2], lines[i][3]));
				}
			}
		}

		//on calcule le rectangle d'intérêt, qui tente de déterminer la zone où se trouve la grille ou on le définit à la souris.
		Vec4i bounds;

		if(!hasBounds)
			bounds = getWeightedBounds(points, points.size()/8, points.size()/8);
		else
			bounds = mouseBounds;

		if(isHorizontal)
		{
			//c'est beau c'est hardcodé
			bounds.val[1]-=30;
		}

		//cout<<"Bounds: "<<bounds<<endl;
		line( color_dst, Point(bounds.val[2], 0), Point(bounds.val[2], img.size().height), Scalar(0, 255, 0), 1, 8 );
		line( color_dst, Point(bounds.val[0], 0), Point(bounds.val[0], img.size().height), Scalar(0, 255, 0), 1, 8 );


		line( color_dst, Point(0, bounds.val[1]), Point(img.size().width, bounds.val[1]), Scalar(0, 255, 0), 1, 8 );
		line( color_dst, Point(0, bounds.val[3]), Point(img.size().width, bounds.val[3]), Scalar(0, 255, 0), 1, 8 );
		//on choisis les lignes les plus à même d'être celles qui définissent le rectangle de la grille, 
		//c'est à dire les plus longues, verticales et qui sont suffisemment près du groupe pour faire partie de la grille

		vector<Vec4i> chosenLines;
		vector<Point2f> chosenPoints;

		int limit = 0;
		for (std::map<float, Vec4i>::iterator i = linesByDistance.begin(); i != linesByDistance.end(); i++)
		{
			if(limit>linesByDistance.size()-linesByDistance.size()/4)
			{
				Point2f p1 = Point2f(i->second.val[0], i->second.val[1]);
				Point2f p2 = Point2f(i->second.val[2], i->second.val[3]);
				if( ((getLineVerticalRatio(p1, p2)>2 && !isHorizontal) || (getLineHorizontalRatio(p1, p2)>2 && isHorizontal))
				&& p1.y>bounds.val[1] 
				&& p1.y<bounds.val[3] 
				&& p1.x>bounds.val[0] 
				&& p1.x<bounds.val[2]

				&& p2.y>bounds.val[1] 
				&& p2.y<bounds.val[3] 
				&& p2.x>bounds.val[0] 
				&& p2.x<bounds.val[2]
				)
				{
					chosenLines.push_back(i->second);
					chosenPoints.push_back(p1);
					chosenPoints.push_back(p2);
					//line(color_dst, p1, p2, Scalar(0, 0, 255), 1);
				}
			}

			limit++;
		}

		//on prends les points les plus proches des 4 extrémités du rectangle d'intérêt, ils devraient définir la grille (ou un truc completement a l'ouest)

		Point2f topLeft;
		Point2f botLeft;
		Point2f topRight;
		Point2f botRight;

		//topLeft (0, 0)
		float bestDistance = 999999999999999;
		for( size_t i = 0; i < chosenPoints.size(); i++ )
		{
			float dist =distance(chosenPoints[i].x, chosenPoints[i].y, bounds.val[0], bounds.val[1]);
			if(dist<bestDistance)
			{
				topLeft = chosenPoints[i];
				bestDistance = dist;
			}
		}

		//botLeft (0, height)
		bestDistance = 999999999999999;
		for( size_t i = 0; i < chosenPoints.size(); i++ )
		{
			float dist =distance(chosenPoints[i].x, chosenPoints[i].y, bounds.val[0], bounds.val[3]);
			if(dist<bestDistance)
			{
				botLeft = chosenPoints[i];
				bestDistance = dist;
			}
		}

		//topRight (width, 0)
		bestDistance = 999999999999999;
		for( size_t i = 0; i < chosenPoints.size(); i++ )
		{
			float dist =distance(chosenPoints[i].x, chosenPoints[i].y, bounds.val[2], bounds.val[1]);
			if(dist<bestDistance)
			{
				topRight = chosenPoints[i];
				bestDistance = dist;
			}
		}
		//botRight (width, height)
		bestDistance = 999999999999999;
		for( size_t i = 0; i < chosenPoints.size(); i++ )
		{
			float dist =distance(chosenPoints[i].x, chosenPoints[i].y, bounds.val[2], bounds.val[3]);
			if(dist<bestDistance)
			{
				botRight = chosenPoints[i];
				bestDistance = dist;
			}
		}
		
		storedPoints.clear();

		if(isHorizontal)
		{
			storedPoints.push_back(botLeft);
			storedPoints.push_back(topLeft);
			storedPoints.push_back(topRight);
			storedPoints.push_back(botRight);
		}
		else
		{
			storedPoints.push_back(topLeft);
			storedPoints.push_back(topRight);
			storedPoints.push_back(botRight);
			storedPoints.push_back(botLeft);
		}

		//dernier test, on s'assure que le quadrialtere final est bien rectangulaire (en prenant en compte la perspective)
		//si le quadrialtere n'est pas rectangulaire, le résultat n'est pas bon, on tente à nouveau de trouver la grille avec d'autres parametres...
		//de plus, plus le quadrialtere s'approche du ratio hauteur/largeur 800x300 plus on est proche de la solution.
		float score;

		if(isHorizontal)
		{
			float defaultRatio = 800.0f/300.0f;
			float currentWidth = (abs(topRight.x-topLeft.x)+abs(botRight.x-botLeft.x))/2.0f;
			float currentHeight = (abs(topRight.x-botRight.x)+abs(topLeft.x-botLeft.x))/2.0f;
			float currentRatio = currentWidth/currentHeight;
			score = abs(topRight.x-botRight.x)+abs(topLeft.x-botLeft.x);
			score*= abs(defaultRatio-currentRatio);
		}
		else
		{
			float defaultRatio = 300.0f/800.0f;
			float currentWidth = (abs(topRight.x-topLeft.x)+abs(botRight.x-botLeft.x))/2.0f;
			float currentHeight = (abs(topRight.x-botRight.x)+abs(topLeft.x-botLeft.x))/2.0f;
			float currentRatio = currentWidth/currentHeight;
			score = abs(topRight.y-topLeft.y)+abs(botRight.y-botLeft.y);
		}
		
		//si le quadirlatere est trop petit il n'est certainement pas viable...
		if(abs(topRight.y-topLeft.y)+abs(botRight.y-botLeft.y)+abs(botRight.x-botLeft.x)+abs(botRight.x-botLeft.x)<50)
			score = 999999;
		
		if(maxLineGap<80)
		{
			if(lowestScore>score)
			{
				bestResult = storedPoints;
				lowestScore = score;
			}

			maxLineGap += 5;

			cout<<"Je reflechis..."<<maxLineGap/80.0f*100.0f<<"%"<<endl;
		}
		else
		{
			//fin du test...
			break;
		}

		//plantage avec HoughLinesP pour le moment on se contente d'une itération.....
		break;
	}

	storedPoints = bestResult;

	circle(color_dst, storedPoints[0], 3, Scalar(255, 255, 0), 8);
	circle(color_dst, storedPoints[1], 3, Scalar(255, 255, 0), 8);
	circle(color_dst, storedPoints[2], 3, Scalar(255, 255, 0), 8);
	circle(color_dst, storedPoints[3], 3, Scalar(255, 255, 0), 8);

	imshow(windowName, color_dst );
}

//détection des réponses a partir d'une grille
void getGridAnswers(string windowName)
{
	//l'image contenant la grille a été isolée et redimentionnée en 300x800, nous pouvons commencer à analyser la grille.
	  cv::Rect myROI(133, 0, cutImage.size().width-133, cutImage.size().height);

	//on découpe ce qui nous intéresse dans la grille.
	cv::Mat croppedImage = cutImage(myROI);

	//on réduit les contrastes pour éviter certains comportements étranges avec des images trop sombres...
	normalizeImage(&croppedImage, 0.3f, 1);

	//on détermine l'intensité de couleur moyenne de l'image, en la résuisant, cela pourra nous servir plus tard...
	Mat reducedImg;
	resize(croppedImage, reducedImg, Size(50, 50));
	float colorRatio = getAverageImageColor(&reducedImg)/134;

	cout<<"colorRatio: "<<colorRatio<<endl;

	//on binarise l'image, lignes par lignes pour maximiser la cohérence des réponses.
	binarizeAndNormalize(&croppedImage, 3);

	//un peu de flou pour aider les tests
	GaussianBlur( croppedImage, croppedImage, Size( 15, 15 ), 0, 0 );
	Mat binaryImg; 
	//on détermine les couleurs min et max
	float max = getMaxColor(&croppedImage);
	float min = getMinColor(&croppedImage);

	float diff = max-min;


	for(float j=0; j<20; j++)
	{
		//on détermine si la ligne est paire, car la grille a un fond qui change de couleur, 
		//cf: inutile grâce à la fonction binarizeAndNormalize qui fait un travail bien plus efficace la dessus. 
		/*
		float oddFactor;

		if ((int)(j) % 2 == 0) {
		 // even
		oddFactor = 1.3;
		} else {
		 // odd
		oddFactor  = 1;
		}
		*/

		for(float i=0; i<4; i++)
		{
			float color;
			color = getAvgPixelColor(&croppedImage, i*40+20, 45+j*37.7+20, 3, 6);
			if(color<max-diff*0.35)
			{
				cout<<"*|";
				circle(croppedImage, Point(i*40+20, 45+j*37.7+20), 3, Scalar(0, 255, 0), 2);
			}
			else
				cout<<"_|";
		}

			cout<<endl;

			imshow(windowName, croppedImage);
	}
}

//on détecte le click souris de cette facon...
void CallBackFunc(int event, int x, int y, int flags, void* userdata)
{
     if  ( event == EVENT_LBUTTONDOWN )
     {
		Mat tmp = img.clone();

		mouseBounds.val[0] = x;
		mouseBounds.val[1] = y;

		if(mouseBounds.val[0]>mouseBounds.val[2])
		mouseBounds.val[0] = mouseBounds.val[2];
		if(mouseBounds.val[1]>mouseBounds.val[3])
		mouseBounds.val[1] = mouseBounds.val[3];

		line( tmp, Point(mouseBounds.val[2], 0), Point(mouseBounds.val[2], img.size().height), Scalar(0, 255, 0), 1, 8 );
		line( tmp, Point(mouseBounds.val[0], 0), Point(mouseBounds.val[0], img.size().height), Scalar(0, 255, 0), 1, 8 );


		line( tmp, Point(0, mouseBounds.val[1]), Point(img.size().width, mouseBounds.val[1]), Scalar(0, 255, 0), 1, 8 );
		line( tmp, Point(0, mouseBounds.val[3]), Point(img.size().width, mouseBounds.val[3]), Scalar(0, 255, 0), 1, 8 );
		imshow("original", tmp);
     }
     else if  ( event == EVENT_RBUTTONDOWN )
     {
		Mat tmp = img.clone();

		mouseBounds.val[2] = x;
		mouseBounds.val[3] = y;

		if(mouseBounds.val[2]<mouseBounds.val[0])
		mouseBounds.val[2] = mouseBounds.val[0];

		if(mouseBounds.val[3]<mouseBounds.val[1])
		mouseBounds.val[3] = mouseBounds.val[1];

		line( tmp, Point(mouseBounds.val[2], 0), Point(mouseBounds.val[2], img.size().height), Scalar(0, 255, 0), 1, 8 );
		line( tmp, Point(mouseBounds.val[0], 0), Point(mouseBounds.val[0], img.size().height), Scalar(0, 255, 0), 1, 8 );


		line( tmp, Point(0, mouseBounds.val[1]), Point(img.size().width, mouseBounds.val[1]), Scalar(0, 255, 0), 1, 8 );
		line( tmp, Point(0, mouseBounds.val[3]), Point(img.size().width, mouseBounds.val[3]), Scalar(0, 255, 0), 1, 8 );
		imshow("original", tmp);
     }
     else if  ( event == EVENT_MBUTTONDOWN )
     {
		findGrid("Lines", true);
		GetImageFromPoints();
     }
     else if ( event == EVENT_MOUSEMOVE )
     {
     }
}

void GetImageFromPoints()
{
	cout<<"Applying transform... "<<endl;

	// On crée la nouvelle image
	cv::Mat quad = cv::Mat::zeros(800, 300, CV_8UC3);

	// Corners of the destination image
	std::vector<cv::Point2f> quad_pts;
	quad_pts.push_back(cv::Point2f(0, 0));
	quad_pts.push_back(cv::Point2f(quad.cols, 0));
	quad_pts.push_back(cv::Point2f(quad.cols, quad.rows));
	quad_pts.push_back(cv::Point2f(0, quad.rows));

	// Get transformation matrix
	cv::Mat transmtx = cv::getPerspectiveTransform(storedPoints, quad_pts);

	// Apply perspective transformation
	cv::warpPerspective(img, quad, transmtx, quad.size());
	cv::imshow("res", quad);

	cutImage = quad;

	getGridAnswers("result");
}

Vec3b getPixelColorAt(Mat* img, int x, int y)
{
	return img->at<Vec3b>(Point(x,y));
}

float getPixelColorAtf(Mat* img, int x, int y)
{
	return img->at<float>(Point(x,y));
}

void setPixelColorAt(Mat* img, int x, int y, uchar R, uchar G, uchar B)
{
	Vec3b color = img->at<Vec3b>(Point(x,y));
	color[0] = R;
	color[1] = G;
	color[2] = B;
	img->at<Vec3b>(Point(x,y)) = color;
}

void setPixelColorAt(Mat* img, int x, int y, float color)
{
	img->at<float>(Point(x,y)) = color;
}

//détermine la valeur R moyenne dans une zone de l'image
float getAvgPixelColor(Mat* img, int x, int y, int step, int range)
{
	float totalVal = 0;

	int division = (range*2+1)*(range*2+1);

	for(int i=-range; i<=range; i++)
	{
		for(int j=-range; j<=range; j++)
		{
			float pixel = img->at<Vec3b>(Point(x+step*i,y+step*j)).val[0];
			pixel += img->at<Vec3b>(Point(x+step*i,y+step*j)).val[1];
			pixel += img->at<Vec3b>(Point(x+step*i,y+step*j)).val[2];
			totalVal += pixel/3;
		}
	}

	totalVal = totalVal/division;

	return totalVal;
}

//détermine la valeur R moyenne de l'image
float getAverageImageColor(Mat* img)
{
	float total=0;
	for(int i = 0; i<img->size().width; i++)
	{
		for(int j=0; j<img->size().height; j++)
		total+=img->at<Vec3b>(Point(i,j)).val[0];
	}

	total = total/(img->size().width*img->size().height);

	return total;
}

//normalise la couleur de l'image pour la faire tendre vers 125.
void normalizeImage(Mat* img, float rate, int iterations)
{
	for(int o=0; o<iterations; o++)
	{
		for(int i = 0; i<img->size().width; i++)
		{
			for(int j=0; j<img->size().height; j++)
			{
				Vec3b pixel =img->at<Vec3b>(Point(i,j));
				pixel.val[0] += (125-pixel.val[0])*rate;
				pixel.val[1] += (125-pixel.val[1])*rate;
				pixel.val[2] += (125-pixel.val[2])*rate;
				img->at<Vec3b>(Point(i,j)) = pixel;
			}
		}
	}
}

//opposé de normalizeImage renforce les contrastes plus ils s'éloignent de 125.
void polarizeImage(Mat* img)
{
	for(int i = 0; i<img->size().width; i++)
	{
		for(int j=0; j<img->size().height; j++)
		{
			Vec3b pixel =img->at<Vec3b>(Point(i,j));
			pixel.val[0] -= (125-pixel.val[0])/2;
			pixel.val[1] -= (125-pixel.val[1])/2;
			pixel.val[2] -= (125-pixel.val[2])/2;

			img->at<Vec3b>(Point(i,j)) = pixel;
		}
	}
}

//renvoie la valeur la plus élevée des couleurs de l'image
int getMaxColor(Mat* img)
{
	int maxColor = 0;
	for(int i = 0; i<img->size().width; i++)
	{
		for(int j=0; j<img->size().height; j++)
		{
			Vec3b pixel =img->at<Vec3b>(Point(i,j));
			if(pixel.val[0]>maxColor)
				maxColor = pixel.val[0];

			if(pixel.val[1]>maxColor)
				maxColor = pixel.val[1];

			if(pixel.val[2]>maxColor)
				maxColor = pixel.val[2];
		}
	}

	return maxColor;
}

//renvoie la valeur la plus basse des couleurs de l'image
int getMinColor(Mat* img)
{
	int minColor = 255;
	for(int i = 0; i<img->size().width; i++)
	{
		for(int j=0; j<img->size().height; j++)
		{
			Vec3b pixel =img->at<Vec3b>(Point(i,j));
			if(pixel.val[0]<minColor)
				minColor = pixel.val[0];

			if(pixel.val[1]<minColor)
				minColor = pixel.val[1];

			if(pixel.val[2]<minColor)
				minColor = pixel.val[2];
		}
	}

	return minColor;
}

//renvoie la valeur la plus haute des couleurs de l'image dans une zone
int getMaxColorInArea(Mat* img, Rect area)
{
	int maxColor = 0;
	for(int i = area.x; i<area.x+area.width; i++)
	{
		for(int j=area.y; j<area.y+area.height; j++)
		{
			if(i>=0 && j>=0 && i<img->size().width && j<img->size().height)
			{
				Vec3b pixel =img->at<Vec3b>(Point(i,j));
				if(pixel.val[0]>maxColor)
					maxColor = pixel.val[0];

				if(pixel.val[1]>maxColor)
					maxColor = pixel.val[1];

				if(pixel.val[2]>maxColor)
					maxColor = pixel.val[2];
			}
		}
	}

	return maxColor;
}

//renvoie la valeur la plus basse des couleurs de l'image dans une zone
int getMinColorInArea(Mat* img, Rect area)
{
	int minColor = 255;
	for(int i = area.x; i<area.x+area.width; i++)
	{
		for(int j=area.y; j<area.y+area.height; j++)
		{
			if(i>=0 && j>=0 && i<img->size().width && j<img->size().height)
			{
				Vec3b pixel =img->at<Vec3b>(Point(i,j));
				if(pixel.val[0]<minColor)
					minColor = pixel.val[0];

				if(pixel.val[1]<minColor)
					minColor = pixel.val[1];

				if(pixel.val[2]<minColor)
					minColor = pixel.val[2];
			}
		}
	}

	return minColor;
}

//convertis les pixels de l'image dans la zone spécifiée en blanc ou noir si elles sont ou pas dans l'interval min/max spécifé (ou inversement si inverted est vrai)
void thresholdArea(Mat* img, Rect area, int min, int max, bool inverted)
{
	for(int i = area.x; i<area.x+area.width; i++)
	{
		for(int j=area.y; j<area.y+area.height; j++)
		{
			if(i>=0 && j>=0 && i<img->size().width && j<img->size().height)
			{
				Vec3b pixel =img->at<Vec3b>(Point(i,j));

				if(!inverted)
				{
					if(pixel.val[0]>min && pixel.val[0]<max)
						pixel.val[0] = 255;
					else
						pixel.val[0] = 0;

					if(pixel.val[1]>min && pixel.val[1]<max)
						pixel.val[1] = 255;
					else
						pixel.val[1] = 0;

					if(pixel.val[2]>min && pixel.val[2]<max)
						pixel.val[2] = 255;
					else
						pixel.val[2] = 0;
				}
				else
				{
					if(pixel.val[0]<min || pixel.val[0]>max)
						pixel.val[0] = 255;
					else
						pixel.val[0] = 0;

					if(pixel.val[1]<min || pixel.val[1]>max)
						pixel.val[1] = 255;
					else
						pixel.val[1] = 0;

					if(pixel.val[2]<min || pixel.val[2]>max)
						pixel.val[2] = 255;
						else
						pixel.val[2] = 0;
				}

				img->at<Vec3b>(Point(i,j)) = pixel;
			}
		}
	}
}

//applique thresholdArea en détectant les valeurs min/max pour chaques tranches (définies dans step)
void binarizeAndNormalize(Mat* img, float tolerance)
{
Vec2i step = Vec2i(img->size().width, 3);

for(int xPos = 0; xPos<img->size().width; xPos+=step.val[0])
{
for(int yPos = 45; yPos<img->size().height; yPos+=step.val[1])
{
Rect area = Rect(xPos, yPos, step.val[0], step.val[1]);

int min = getMinColorInArea(img, area);
int max = getMaxColorInArea(img, area);

int diff = max-min;

thresholdArea(img, area, min+diff/8, max-diff/tolerance, true);
}
}
}

void binarizeAndNormalize(Mat* img, float tolerance, Vec2i step)
{
for(int xPos = 0; xPos<img->size().width; xPos+=step.val[0])
{
for(int yPos = 0; yPos<img->size().height; yPos+=step.val[1])
{
Rect area = Rect(xPos, yPos, step.val[0], step.val[1]);

int min = getMinColorInArea(img, area);
int max = getMaxColorInArea(img, area);

int diff = max-min;

thresholdArea(img, area, min+diff/8, max-diff/tolerance, true);
}
}
}

double CCW(Vec2f a, Vec2f b, Vec2f c)
{ 
return (b.val[0]-a.val[0])*(c.val[1]-a.val[1]) - (b.val[1]-a.val[1])*(c.val[0]-a.val[0]); 
}

int middle(int a, int b, int c) {
  int t;    
  if ( a > b ) {
    t = a;
    a = b;
    b = t;
  }
  if ( a <= c && c <= b ) return 1;
  return 0;
}

int intersect(line2 a, line2 b) {
  if ( ( CCW(a.s, a.e, b.s) * CCW(a.s, a.e, b.e) < 0 ) &&
     ( CCW(b.s, b.e, a.s) * CCW(b.s, b.e, a.e) < 0 ) ) return 1;

  if ( CCW(a.s, a.e, b.s) == 0 && middle(a.s.val[0], a.e.val[0], b.s.val[0]) && middle(a.s.val[1], a.e.val[1], b.s.val[1]) ) return 1;
  if ( CCW(a.s, a.e, b.e) == 0 && middle(a.s.val[0], a.e.val[0], b.e.val[0]) && middle(a.s.val[1], a.e.val[1], b.e.val[1]) ) return 1;
  if ( CCW(b.s, b.e, a.s) == 0 && middle(b.s.val[0], b.e.val[0], a.s.val[0]) && middle(b.s.val[1], b.e.val[1], a.s.val[1]) ) return 1;
  if ( CCW(b.s, b.e, a.e) == 0 && middle(b.s.val[0], b.e.val[0], a.e.val[0]) && middle(b.s.val[1], b.e.val[1], a.e.val[1]) ) return 1;

    return 0;
}

float distance(float p1x, float p1y, float p2x, float p2y)
{
    float distance = sqrt( (p1x - p2x) * (p1x - p2x) +
                            (p1y - p2y) * (p1y - p2y) );
    return distance;
}

//cette methode permet de définir un rectangle tracé par la majorité des points (une sorte de barycentre avec 4 extremités)
Vec4i getWeightedBounds(vector<Point> points, float toleranceX, float toleranceY)
{
	//On commence par centrer les points initiaux.
	float maxX = img.size().width/2;
	float minX = img.size().width/2;
	float maxY = img.size().height/2;
	float minY = img.size().height/2;

	float fact = points.size()/16;

	//On applique la magie!
	for( size_t i = 1; i < points.size(); i++ )
	{
		if(points[i].x>maxX)
			maxX+=abs(float(points[i].x-img.size().width))/toleranceX;

		if(points[i].x<minX)
			minX-=abs(float(points[i].x))/fact;

		if(points[i].y>maxY)
			maxY+=abs(float(points[i].y-img.size().height))/toleranceY;

		if(points[i].y<minY)
			minY-=abs(float(points[i].y))/fact;
	}
	return Vec4i(minX, minY, maxX, maxY);
}

float getLineVerticalRatio(Point A, Point B)
{
	float horizontal = abs(A.x-B.x);
	float vertical = abs(A.y-B.y);

	if(horizontal==0)
	return 99999;
	else
	return vertical/horizontal;
}

float getLineHorizontalRatio(Point A, Point B)
{
	float horizontal = abs(A.x-B.x);
	float vertical = abs(A.y-B.y);

	if(vertical==0)
	return 99999;
	else
	return horizontal/vertical;
}