#include <opencv2/core/core.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <vector>
#include <algorithm>
#include <math.h>
#include <time.h>
#include <string.h>
#include <ctime>

using namespace std;
using namespace cv;
clock_t start, stop;
int num=0;

class LaneDetect
{
public:
    Mat currFrame; //stores the upcoming frame,mat means 矩阵
    Mat temp;      //stores intermediate results
    Mat temp2;     //stores the final lane segments

    int diff, diffL, diffR;
    int laneWidth;
    int diffThreshTop;
    int diffThreshLow;
    int ROIrows;
    int vertical_left;//vertical=垂直
    int vertical_right;
    int vertical_top;
    int smallLaneArea;
    int longLane;
    int  vanishingPt;
    float maxLaneWidth;

    //to store various blob properties
    Mat binary_image; //used for blob removal
    int minSize;
    int ratio;
    float  contour_area;
    float blob_angle_deg;
    float bounding_width;
    float bounding_length;
    Size2f sz;//size,2dims,float
    vector< vector<Point> > contours; //浮点类型的轮廓。坐标
    vector<Vec4i> hierarchy; //固定写法，后来findcontours算出边界的坐标，存在contours里面。
    RotatedRect rotated_rect; //填充


    LaneDetect(Mat startFrame)
    {
        //currFrame = startFrame;                                    //if image has to be processed at original size

        currFrame = Mat(320,480,CV_8UC1,0.0);                        //initialised the image size to 320x480  CV_8UC1---则可以创建----8位无符号整形的单通道---灰度图片
        resize(startFrame, currFrame, currFrame.size());             // resize the input to required size

        temp      = Mat(currFrame.rows, currFrame.cols, CV_8UC1,0.0);//stores possible lane markings，rows行，cols列
        temp2     = Mat(currFrame.rows, currFrame.cols, CV_8UC1,0.0);//stores finally selected lane marks

        vanishingPt    = currFrame.rows/2;                           //for simplicity right now //分母变大，增加天空轮廓，分母变小，只认路面
        ROIrows        = currFrame.rows - vanishingPt;               //rows in region of interest
        minSize        = 0.0015 * (currFrame.cols*currFrame.rows);  //min size of any region to be selected as lane //数值为23左右
        maxLaneWidth   = 0.025 * currFrame.cols;                     //approximate max lane width based on image size 宽度为12列左右
        smallLaneArea  = 7 * minSize;
        longLane       = 0.7 * currFrame.rows;
        ratio          = 4;

        //these mark the possible ROI for vertical lane segments and to filter vehicle glare
        vertical_left  = 2*currFrame.cols/5;
        vertical_right = 3*currFrame.cols/5;
        vertical_top   = 2*currFrame.rows/3;

        namedWindow("lane",3);
        namedWindow("midstep", 3);
        namedWindow("currframe", 3); //name,数字是窗口size;
        namedWindow("laneBlobs",3);
		namedWindow("NoParkingArea",5);

        getLane();
    }

    void updateSensitivity()
    {
        int total=0, average =0;
        for(int i= vanishingPt; i<currFrame.rows; i++)
            for(int j= 0 ; j<currFrame.cols; j++)
                total += currFrame.at<uchar>(i,j);//total=total+第（i，j）的灰度，计算160*480的灰度和
        average = total/(ROIrows*currFrame.cols);
        cout<<"average : "<<average<<endl;
    }

    void getLane()
    {
        //medianBlur(currFrame, currFrame,5 );
        // updateSensitivity();
        //ROI = bottom half
        for(int i=vanishingPt; i<currFrame.rows; i++)
            for(int j=0; j<currFrame.cols; j++)
            {
                temp.at<uchar>(i,j)    = 0;
                temp2.at<uchar>(i,j)   = 0;//temp，temp2的图像，160*480灰度变成0，右半边变成0
            }

        imshow("currframe", currFrame);
        blobRemoval();
    }
	void int2str(const int &int_temp,string &string_temp)  
	{  
			stringstream stream;  
			stream<<int_temp;  
			string_temp=stream.str();   //此处也可以用 stream>>string_temp  
	}  
    void markLane()
    {
        for(int i=vanishingPt; i<currFrame.rows; i++)
        {
            //IF COLOUR IMAGE IS GIVEN then additional check can be done
            // lane markings RGB values will be nearly same to each other(i.e without any hue)

            //min lane width is taken to be 5
            laneWidth =5+ maxLaneWidth*(i-vanishingPt)/ROIrows;
            for(int j=laneWidth; j<currFrame.cols- laneWidth; j++)
            {

                diffL = currFrame.at<uchar>(i,j) - currFrame.at<uchar>(i,j-laneWidth);
                diffR = currFrame.at<uchar>(i,j) - currFrame.at<uchar>(i,j+laneWidth);//（160.5）与（160.0）,（160.10）的差值，，（160,6）与（160,1），（160,11）中间点与两边的差值。。。。。。。多次循环，遍历大多数基本所有像素。
				//总体是计算某点与其水平临近像素方向之差
                diff  =  diffL + diffR - abs(diffL-diffR);//中间点与两边的差值求和，减去差值之差

                //1 right bit shifts to make it 0.5 times
                diffThreshLow = currFrame.at<uchar>(i,j)>>1; //右移一位，除以2，（）下半边都除2）
                //diffThreshTop = 1.2*currFrame.at<uchar>(i,j);

                //both left and right differences can be made to contribute
                //at least by certain threshold (which is >0 right now)
                //total minimum Diff should be atleast more than 5 to avoid noise
                if (diffL>0 && diffR >0 && diff>5)
                    if(diff>=diffThreshLow /*&& diff<= diffThreshTop*/ )
                        temp.at<uchar>(i,j)=255;//根据像素差值找到白线
            }
        }

	
		
		//for (int i = 0; i < 479; i++) 
	//	{
		//	cout<<"uchar="<<temp.at<uchar>(160,i)<<endl;
		//	if (temp.at<uchar>(161, i) >= 200);
			//{
			//	sum_row_min += i;
			//	num_min = num_min + 1;
			//}
			//if (temp.at<uchar>(319, i) >= 200)
			//{
			//	sum_row_max += i;
			//	num_max = num_max + 1;
			//}
			//else
			//{
				//
		//	}
			

	//	}
		//avr_row_min = sum_row_min / num_min;
		//avr_row_max = sum_row_max / num_max;
		//cout << "location of min row is" << avr_row_min << endl;
		//cout << "location of max row is" << avr_row_max << endl;

    }

    void blobRemoval()
    {
	
        markLane();
	static int num=0;
	string name;
	time_t rawtime;
	struct tm *info;
	char buffer[128];
	time(&rawtime);
	info =localtime(&rawtime);
	strftime(buffer,80,"%b%d_%H%M%S",info);
        string ss1=string(buffer);
	//ss1=buffer;
	//cout<<ss1<<endl;
	cout<<ss1<<endl;	


        // find all contours in the binary image
        temp.copyTo(binary_image);
        findContours(binary_image, contours,
                     hierarchy, CV_RETR_CCOMP,
                     CV_CHAIN_APPROX_SIMPLE);
		//固定用法//可能需要在这里找到坐标点，https://blog.csdn.net/dcrmg/article/details/51987348//
		//CCOMP检测所有轮廓，但是只建立2个登记关系，外围为顶层，若外围轮廓包含其他轮廓信息，内为轮廓也保存为顶层。
		//SIMPLE仅保存轮廓拐点信息。
        // for removing invalid blobs
        if (!contours.empty()) //有边界就向下执行
        {
            for (size_t i=0; i<contours.size(); ++i)
            {
                //====conditions for removing contours====//

                contour_area = contourArea(contours[i]) ;

                //blob size should not be less than lower threshold
                if(contour_area > minSize)
                {
                    rotated_rect    = minAreaRect(contours[i]);
                    sz              = rotated_rect.size;
                    bounding_width  = sz.width;
                    bounding_length = sz.height;


                    //openCV selects length and width based on their orientation
                    //so angle needs to be adjusted accordingly
                    blob_angle_deg = rotated_rect.angle;
                    if (bounding_width < bounding_length)
                        blob_angle_deg = 90 + blob_angle_deg;   //转换角度，将width与height互换。扁矩形变成长矩形。以X轴为限，逆时针为负角度，顺时针为正角度。所以加90度可以调换矩形方位与宽高

                    //if such big line has been detected then it has to be a (curved or a normal)lane
                    if(bounding_length>longLane || bounding_width >longLane)
                    {
                        drawContours(currFrame, contours,i, Scalar(255), CV_FILLED, 8);  //对drawContours操作
                        drawContours(temp2, contours,i, Scalar(255), CV_FILLED, 8);
                    }

                    //angle of orientation of blob should not be near horizontal or vertical
                    //vertical blobs are allowed only near center-bottom region, where centre lane mark is present
                    //length:width >= ratio for valid line segments
                    //if area is very small then ratio limits are compensated
                    else if ((blob_angle_deg <-10 || blob_angle_deg >-10 ) &&
                             ((blob_angle_deg > -70 && blob_angle_deg < 70 ) ||
                              (rotated_rect.center.y > vertical_top &&
                               rotated_rect.center.x > vertical_left && rotated_rect.center.x < vertical_right)))   //使角度不至于过大或过小，趋近于X或Y轴
                    {

                        if ((bounding_length/bounding_width)>=ratio || (bounding_width/bounding_length)>=ratio
                                ||(contour_area< smallLaneArea &&  ((contour_area/(bounding_width*bounding_length)) > .75) &&
                                   ((bounding_length/bounding_width)>=2 || (bounding_width/bounding_length)>=2)))
                        {
                            drawContours(currFrame, contours,i, Scalar(255), CV_FILLED, 8);
                            drawContours(temp2, contours,i, Scalar(255), CV_FILLED, 8);
							//imshow("123",temp2);
							//waitKey(9999);
                        }
                    }
                }
            }
        }
		int sum_row_min = 0;
		int sum_row_max = 0;
		int avr_row_min;
		int avr_row_max;
		int num_min = 0;
		int num_max = 0;
		float avr;
	        int num_save=0;
	
		//Mat noparking(320,480,CV_8UC1,128);
		Mat noparking;
		for (int i = 0; i < currFrame.cols; i++) 
		{
			if (temp.at<uchar>(200, i) >= 200||temp.at<uchar>(210, i) >= 200||temp.at<uchar>(220, i) >= 200||temp.at<uchar>(195, i) >= 200||temp.at<uchar>(190, i) >= 200)
			//{
				//cout<<temp2.at<uchar>(180,i)<<endl;
				//cout<<i<<endl;
				sum_row_min += i;
				num_min = num_min + 1;
			//}
		}
		for(int j=0;j<currFrame.cols;j++)
		{
			//cout<<temp2.at<uchar>(320,j)<<endl;
			if (temp.at<uchar>(320, j) >= 200||temp.at<uchar>(310, j) >= 200||temp.at<uchar>(315, j) >= 200)
			{
				sum_row_max += j;
				num_max = num_max + 1;
				//cout<<max(j)<<endl;
			}

		}
                
	//	Point root_points[1][4];
	//	root_points[0][0]=Point(160,avr_row_min);
	//	root_points[0][1]=Point(320,avr_row_max);
	//	root_points[0][2]=Point(160,480);
	//	root_points[0][3]=Point(320,480);


		
	//	int y_max=avr_row_max;
		//cout<<sum_row_min<<endl;
		//cout<<sum_row_max<<endl;
		int num_1=0;
		//cout<<sum_row_max<<endl;
		
		avr_row_min = sum_row_min / num_min;
		avr_row_max = sum_row_max / num_max;
		avr=(avr_row_min+avr_row_max)/1.41;
		cout << "location of min row is" << avr_row_min << endl;
		cout << "location of max row is" << avr_row_max << endl;
		cout<<"avr is "<<avr<<endl;
	//	Rect rect1(170,avr_row_min,450-avr_row_min,140);
	 //   temp2(rect1).copyTo(noparking);
		Rect rect2(avr,160,480-avr,160);
		currFrame(rect2).copyTo(noparking);
		//sprintf(ori,"s%d%s%","ori",++num,".jpg");
		char ori[128];
		char nopark[32];
	    //sprintf(nopark,"s%d%s%","nopark",num,".jpg");
		num++;
		//string count=str(num);
		//std::string name = std::to_string(num);
	        int2str(num,name);
                if(num%15==0)
	       {
	        imwrite("/home/sdu/lanedetection/no_parking/"+ss1+".jpg",noparking);
               // imwrite(name+".jpg",noparking);
                imwrite("/home/sdu/lanedetection/original/"+ss1+"ori.jpg",currFrame);
               }
		if(num%15==1)
		{
	        imwrite("/home/sdu/lanedetection/no_parking/"+ss1+"_"+".jpg",noparking);
               // imwrite(name+".jpg",noparking);
                imwrite("/home/sdu/lanedetection/original/"+ss1+"_"+"ori.jpg",currFrame);
               }
                //if(num%16==0)
              // {
               // imwrite(name+".jpg",frame);
              // }
                num_save++;
	//waitKey(10000);
		//cout<<num<<endl;
	//	cout<<name<<endl;
		imshow("NoParkingArea",noparking);
        imshow("midstep", temp);
        imshow("laneBlobs", temp2);
        imshow("lane",currFrame);

	//	nextFrame;


    }


    void nextFrame(Mat &nxt)
    {
        //currFrame = nxt;                        //if processing is to be done at original size

        resize(nxt ,currFrame, currFrame.size()); //resizing the input image for faster processing
        getLane();
    }

    Mat getResult()
    {
        return temp2;
    }

};//end of class LaneDetect


void makeFromVid(string path)
{
	
    Mat frame;
    VideoCapture cap(path); // open the video file for reading

    if ( !cap.isOpened() )  // if not success, exit program
        cout << "Cannot open the video file" << endl;

    //cap.set(CV_CAP_PROP_POS_MSEC, 300); //start the video at 300ms

    double fps = cap.get(CV_CAP_PROP_FPS); //get the frames per seconds of the video
    cout << "Input video's Frame per seconds : " << fps << endl;

    cap.read(frame);

    LaneDetect detect(frame);

    while(1)
    {
        bool bSuccess = cap.read(frame); // read a new frame from video
        if (!bSuccess)                   //if not success, break loop
        {
            cout << "Cannot read the frame from video file" << endl;
            exit(0);
            break;
            
        }
	    namedWindow("ori",5);
	    imshow("ori",frame);
        cvtColor(frame, frame, CV_BGR2GRAY);

        //start = clock();
        detect.nextFrame(frame);
        //stop =clock();
        // cout<<"fps : "<<1.0/(((double)(stop-start))/ CLOCKS_PER_SEC)<<endl;

        if(waitKey(10) == 27) //wait for 'esc' key press for 10 ms. If 'esc' key is pressed, break loop
        {
            cout<<"video paused!, press q to quit, any other key to continue"<<endl;
            if(waitKey(0) == 'q')
            {
                cout << "terminated by user" << endl;
                break;
            }
        }
    }
}

int main()
{
	
   makeFromVid("/home/sdu/6.MP4");
    // makeFromVid("/home/yash/opencv-2.4.10/programs/road.m4v");
    waitKey(0);
    destroyAllWindows();
}
