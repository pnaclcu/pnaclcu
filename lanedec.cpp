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
    Mat currFrame; //stores the upcoming frame,mat means ����
    Mat temp;      //stores intermediate results
    Mat temp2;     //stores the final lane segments

    int diff, diffL, diffR;
    int laneWidth;
    int diffThreshTop;
    int diffThreshLow;
    int ROIrows;
    int vertical_left;//vertical=��ֱ
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
    vector< vector<Point> > contours; //�������͵�����������
    vector<Vec4i> hierarchy; //�̶�д��������findcontours����߽�����꣬����contours���档
    RotatedRect rotated_rect; //���


    LaneDetect(Mat startFrame)
    {
        //currFrame = startFrame;                                    //if image has to be processed at original size

        currFrame = Mat(320,480,CV_8UC1,0.0);                        //initialised the image size to 320x480  CV_8UC1---����Դ���----8λ�޷������εĵ�ͨ��---�Ҷ�ͼƬ
        resize(startFrame, currFrame, currFrame.size());             // resize the input to required size

        temp      = Mat(currFrame.rows, currFrame.cols, CV_8UC1,0.0);//stores possible lane markings��rows�У�cols��
        temp2     = Mat(currFrame.rows, currFrame.cols, CV_8UC1,0.0);//stores finally selected lane marks

        vanishingPt    = currFrame.rows/2;                           //for simplicity right now //��ĸ������������������ĸ��С��ֻ��·��
        ROIrows        = currFrame.rows - vanishingPt;               //rows in region of interest
        minSize        = 0.0015 * (currFrame.cols*currFrame.rows);  //min size of any region to be selected as lane //��ֵΪ23����
        maxLaneWidth   = 0.025 * currFrame.cols;                     //approximate max lane width based on image size ���Ϊ12������
        smallLaneArea  = 7 * minSize;
        longLane       = 0.7 * currFrame.rows;
        ratio          = 4;

        //these mark the possible ROI for vertical lane segments and to filter vehicle glare
        vertical_left  = 2*currFrame.cols/5;
        vertical_right = 3*currFrame.cols/5;
        vertical_top   = 2*currFrame.rows/3;

        namedWindow("lane",3);
        namedWindow("midstep", 3);
        namedWindow("currframe", 3); //name,�����Ǵ���size;
        namedWindow("laneBlobs",3);
		namedWindow("NoParkingArea",5);

        getLane();
    }

    void updateSensitivity()
    {
        int total=0, average =0;
        for(int i= vanishingPt; i<currFrame.rows; i++)
            for(int j= 0 ; j<currFrame.cols; j++)
                total += currFrame.at<uchar>(i,j);//total=total+�ڣ�i��j���ĻҶȣ�����160*480�ĻҶȺ�
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
                temp2.at<uchar>(i,j)   = 0;//temp��temp2��ͼ��160*480�Ҷȱ��0���Ұ�߱��0
            }

        imshow("currframe", currFrame);
        blobRemoval();
    }
	void int2str(const int &int_temp,string &string_temp)  
	{  
			stringstream stream;  
			stream<<int_temp;  
			string_temp=stream.str();   //�˴�Ҳ������ stream>>string_temp  
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
                diffR = currFrame.at<uchar>(i,j) - currFrame.at<uchar>(i,j+laneWidth);//��160.5���루160.0��,��160.10���Ĳ�ֵ������160,6���루160,1������160,11���м�������ߵĲ�ֵ�����������������ѭ������������������������ء�
				//�����Ǽ���ĳ������ˮƽ�ٽ����ط���֮��
                diff  =  diffL + diffR - abs(diffL-diffR);//�м�������ߵĲ�ֵ��ͣ���ȥ��ֵ֮��

                //1 right bit shifts to make it 0.5 times
                diffThreshLow = currFrame.at<uchar>(i,j)>>1; //����һλ������2�������°�߶���2��
                //diffThreshTop = 1.2*currFrame.at<uchar>(i,j);

                //both left and right differences can be made to contribute
                //at least by certain threshold (which is >0 right now)
                //total minimum Diff should be atleast more than 5 to avoid noise
                if (diffL>0 && diffR >0 && diff>5)
                    if(diff>=diffThreshLow /*&& diff<= diffThreshTop*/ )
                        temp.at<uchar>(i,j)=255;//�������ز�ֵ�ҵ�����
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
		//�̶��÷�//������Ҫ�������ҵ�����㣬https://blog.csdn.net/dcrmg/article/details/51987348//
		//CCOMP�����������������ֻ����2���Ǽǹ�ϵ����ΧΪ���㣬����Χ������������������Ϣ����Ϊ����Ҳ����Ϊ���㡣
		//SIMPLE�����������յ���Ϣ��
        // for removing invalid blobs
        if (!contours.empty()) //�б߽������ִ��
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
                        blob_angle_deg = 90 + blob_angle_deg;   //ת���Ƕȣ���width��height����������α�ɳ����Ρ���X��Ϊ�ޣ���ʱ��Ϊ���Ƕȣ�˳ʱ��Ϊ���Ƕȡ����Լ�90�ȿ��Ե������η�λ����

                    //if such big line has been detected then it has to be a (curved or a normal)lane
                    if(bounding_length>longLane || bounding_width >longLane)
                    {
                        drawContours(currFrame, contours,i, Scalar(255), CV_FILLED, 8);  //��drawContours����
                        drawContours(temp2, contours,i, Scalar(255), CV_FILLED, 8);
                    }

                    //angle of orientation of blob should not be near horizontal or vertical
                    //vertical blobs are allowed only near center-bottom region, where centre lane mark is present
                    //length:width >= ratio for valid line segments
                    //if area is very small then ratio limits are compensated
                    else if ((blob_angle_deg <-10 || blob_angle_deg >-10 ) &&
                             ((blob_angle_deg > -70 && blob_angle_deg < 70 ) ||
                              (rotated_rect.center.y > vertical_top &&
                               rotated_rect.center.x > vertical_left && rotated_rect.center.x < vertical_right)))   //ʹ�ǶȲ����ڹ�����С��������X��Y��
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
