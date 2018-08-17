                                       // This is a demo code for using a SSD model to do detection.
// The code is modified from examples/cpp_classification/classification.cpp.
// Usage:
//    ssd_detect [FLAGS] model_file weights_file list_file
//
// where model_file is the .prototxt file defining the network architecture, and
// weights_file is the .caffemodel file containing the network parameters, and
// list_file contains a list of image files with the format as follows:
//    folder/img1.JPEG
//    folder/img2.JPEG
// list_file can also contain a list of video files with the format as follows:
//    folder/video1.mp4
//    folder/video2.mp4
/////
#include <caffe/caffe.hpp>
#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif  // USE_OPENCV
#include <algorithm>
#include <iomanip>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <stdio.h>
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

#ifdef USE_OPENCV
using namespace caffe;
using namespace cv;
using namespace std;
Mat img_ori;
void int2str(const int &int_temp,string &string_temp)  
	{  
			stringstream stream;  
			stream<<int_temp;  
			string_temp=stream.str();   //此处也可以用 stream>>string_temp  
	}  
clock_t start, stop; // NOLINT(build/namespaces)
int sum_row_min = 0;
		int sum_row_max = 0;
		int avr_row_min;
		int avr_row_max;
		int num_min = 1;
		int num_max = 1;
		int avr=0;
	        int num_save=0;
Mat noparking;
//Mat temp2;


class LaneDetect
{
public:
    Mat currFrame; //stores the upcoming frame,mat means 
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

     //   namedWindow("lane",3);
        //namedWindow("midstep", 3);
        namedWindow("currframe", 3); //name,数字是窗口size;
       // namedWindow("laneBlobs",3);
		//namedWindow("NoParkingArea",5);

        getLane();
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
	//std::cout<<ss1<<std::endl;	
       
        //cvtColor(currFrame,currFrame,CV_BGR2GRAY);
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
		//int sum_row_min = 0;
		//int sum_row_max = 0;
		//int avr_row_min;
		//int avr_row_max;
		//int num_min = 1;
		//int num_max = 1;
		//int avr=0;
	        //int num_save=0;
	
		//Mat noparking(320,480,CV_8UC1,128);
		//Mat noparking;
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
		//int num_1=0;
		//cout<<sum_row_max<<endl;
		
		avr_row_min = sum_row_min / num_min;
		avr_row_max = sum_row_max / num_max;
                avr=(avr_row_min+avr_row_max)/1;
                int avr2=(avr_row_min+avr_row_max)/2.1;
                int avr3=(avr_row_min+avr_row_max)/2.4;
                Point p1(avr_row_min+30,161);
                Point p2(avr3,200);
                Point p3(avr2,240);
                Point p4(avr,280);
                Point p5(avr_row_max,320);
              //  line(currFrame,p1,p2,Scalar(255,255,255),3);
               // line(currFrame,p2,p3,Scalar(255,255,255),3);
	       // line(currFrame,p3,p4,Scalar(255,255,255),3);
                // line(currFrame,p4,p5,Scalar(255,255,255),3); 
		cout << "location of min row is" << avr_row_min << endl;
		cout << "location of max row is" << avr_row_max << endl;
		cout<<"avr is "<<avr<<endl;
	//	Rect rect1(170,avr_row_min,450-avr_row_min,140);
	 //   temp2(rect1).copyTo(noparking);
		Rect rect2(avr,160,480-avr,160);
		currFrame(rect2).copyTo(noparking);
		//sprintf(ori,"s%d%s%","ori",++num,".jpg");
		//char ori[128];
		//char nopark[32];
	    //sprintf(nopark,"s%d%s%","nopark",num,".jpg");
		num++;
		//string count=str(num);
		//std::string name = std::to_string(num);
	        int2str(num,name);
                if(num%3==0)
	       {
	        imwrite("/home/sdu/lanedetection/no_parking/"+ss1+".jpg",noparking);
               // imwrite(name+".jpg",noparking);
                imwrite("/home/sdu/lanedetection/original/"+ss1+"ori.jpg",currFrame);
               }
		//if(num%15==1)
		//{
	        //imwrite("/home/sdu/lanedetection/no_parking/"+ss1+"_"+".jpg",noparking);
               // imwrite(name+".jpg",noparking);
                //imwrite("/home/sdu/lanedetection/original/"+ss1+"_"+"ori.jpg",currFrame);
              // }
                //if(num%16==0)
              // {
               // imwrite(name+".jpg",frame);
              // }
                num_save++;
	//waitKey(10000);
		//cout<<num<<endl;
	//	cout<<name<<endl;
	//imshow("NoParkingArea",noparking);
       // imshow("midstep", temp);
      //  imshow("laneBlobs", temp2);
       // imshow("lane",currFrame);

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

};

class Detector {
 public:
  Detector(const string& model_file,
           const string& weights_file,
           const string& mean_file,
           const string& mean_value);

  std::vector<vector<float> > Detect(const cv::Mat& img);


 private:
  void SetMean(const string& mean_file, const string& mean_value);

  void WrapInputLayer(std::vector<cv::Mat>* input_channels);

  void Preprocess(const cv::Mat& img,
                  std::vector<cv::Mat>* input_channels);

 private:
  shared_ptr<Net<float> > net_;
  cv::Size input_geometry_;
  int num_channels_;
  cv::Mat mean_;
};

Detector::Detector(const string& model_file,
                   const string& weights_file,
                   const string& mean_file,
                   const string& mean_value) {
#ifdef CPU_ONLY
  Caffe::set_mode(Caffe::CPU);
#else
  Caffe::set_mode(Caffe::GPU);
#endif

  /* Load the network. */
  net_.reset(new Net<float>(model_file, TEST));
  net_->CopyTrainedLayersFrom(weights_file);

  CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";
  CHECK_EQ(net_->num_outputs(), 1) << "Network should have exactly one output.";

  Blob<float>* input_layer = net_->input_blobs()[0];
  num_channels_ = input_layer->channels();
  CHECK(num_channels_ == 3 || num_channels_ == 1)
    << "Input layer should have 1 or 3 channels.";
  input_geometry_ = cv::Size(input_layer->width(), input_layer->height());

  /* Load the binaryproto mean file. */
  SetMean(mean_file, mean_value);
}

std::vector<vector<float> > Detector::Detect(const cv::Mat& img) {
  Blob<float>* input_layer = net_->input_blobs()[0];
  input_layer->Reshape(1, num_channels_,
                       input_geometry_.height, input_geometry_.width);
  /* Forward dimension change to all layers. */
  net_->Reshape();

  std::vector<cv::Mat> input_channels;
  WrapInputLayer(&input_channels);

  Preprocess(img, &input_channels);

  net_->Forward();

  /* Copy the output layer to a std::vector */
  Blob<float>* result_blob = net_->output_blobs()[0];
  const float* result = result_blob->cpu_data();
  const int num_det = result_blob->height();
  vector<vector<float> > detections;
  for (int k = 0; k < num_det; ++k) {
    if (result[0] == -1) {
      // Skip invalid detection.
      result += 7;
      continue;
    }
    vector<float> detection(result, result + 7);
    detections.push_back(detection);
    result += 7;
  }
  return detections;
}

/* Load the mean file in binaryproto format. */
void Detector::SetMean(const string& mean_file, const string& mean_value) {
  cv::Scalar channel_mean;
  if (!mean_file.empty()) {
    CHECK(mean_value.empty()) <<
      "Cannot specify mean_file and mean_value at the same time";
    BlobProto blob_proto;
    ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);

    /* Convert from BlobProto to Blob<float> */
    Blob<float> mean_blob;
    mean_blob.FromProto(blob_proto);
    CHECK_EQ(mean_blob.channels(), num_channels_)
      << "Number of channels of mean file doesn't match input layer.";

    /* The format of the mean file is planar 32-bit float BGR or grayscale. */
    std::vector<cv::Mat> channels;
    float* data = mean_blob.mutable_cpu_data();
    for (int i = 0; i < num_channels_; ++i) {
      /* Extract an individual channel. */
      cv::Mat channel(mean_blob.height(), mean_blob.width(), CV_32FC1, data);
      channels.push_back(channel);
      data += mean_blob.height() * mean_blob.width();
    }

    /* Merge the separate channels into a single image. */
    cv::Mat mean;
    cv::merge(channels, mean);

    /* Compute the global mean pixel value and create a mean image
     * filled with this value. */
    channel_mean = cv::mean(mean);
    mean_ = cv::Mat(input_geometry_, mean.type(), channel_mean);
  }
  if (!mean_value.empty()) {
    CHECK(mean_file.empty()) <<
      "Cannot specify mean_file and mean_value at the same time";
    stringstream ss(mean_value);
    vector<float> values;
    string item;
    while (getline(ss, item, ',')) {
      float value = std::atof(item.c_str());
      values.push_back(value);
    }
    CHECK(values.size() == 1 || values.size() == num_channels_) <<
      "Specify either 1 mean_value or as many as channels: " << num_channels_;

    std::vector<cv::Mat> channels;
    for (int i = 0; i < num_channels_; ++i) {
      /* Extract an individual channel. */
      cv::Mat channel(input_geometry_.height, input_geometry_.width, CV_32FC1,
          cv::Scalar(values[i]));
      channels.push_back(channel);
    }
    cv::merge(channels, mean_);
  }
}

/* Wrap the input layer of the network in separate cv::Mat objects
 * (one per channel). This way we save one memcpy operation and we
 * don't need to rely on cudaMemcpy2D. The last preprocessing
 * operation will write the separate channels directly to the input
 * layer. */
void Detector::WrapInputLayer(std::vector<cv::Mat>* input_channels) {
  Blob<float>* input_layer = net_->input_blobs()[0];

  int width = input_layer->width();
  int height = input_layer->height();
  float* input_data = input_layer->mutable_cpu_data();
  for (int i = 0; i < input_layer->channels(); ++i) {
    cv::Mat channel(height, width, CV_32FC1, input_data);
    input_channels->push_back(channel);
    input_data += width * height;
  }
}

void Detector::Preprocess(const cv::Mat& img,
                            std::vector<cv::Mat>* input_channels) {
  /* Convert the input image to the input image format of the network. */
  cv::Mat sample;
  if (img.channels() == 3 && num_channels_ == 1)
    cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
  else if (img.channels() == 4 && num_channels_ == 1)
    cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
  else if (img.channels() == 4 && num_channels_ == 3)
    cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
  else if (img.channels() == 1 && num_channels_ == 3)
    cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
  else
    sample = img;

  cv::Mat sample_resized;
  if (sample.size() != input_geometry_)
    cv::resize(sample, sample_resized, input_geometry_);
  else
    sample_resized = sample;

  cv::Mat sample_float;
  if (num_channels_ == 3)
    sample_resized.convertTo(sample_float, CV_32FC3);
  else
    sample_resized.convertTo(sample_float, CV_32FC1);

  cv::Mat sample_normalized;
  cv::subtract(sample_float, mean_, sample_normalized);

  /* This operation will write the separate BGR planes directly to the
   * input layer of the network because it is wrapped by the cv::Mat
   * objects in input_channels. */
  cv::split(sample_normalized, *input_channels);

  CHECK(reinterpret_cast<float*>(input_channels->at(0).data)
        == net_->input_blobs()[0]->cpu_data())
    << "Input channels are not wrapping the input layer of the network.";
}

DEFINE_string(mean_file, "",
    "The mean file used to subtract from the input image.");
DEFINE_string(mean_value, "104,117,123",
    "If specified, can be one value or can be same as image channels"
    " - would subtract from the corresponding channel). Separated by ','."
    "Either mean_file or mean_value should be provided, not both.");
DEFINE_string(file_type, "image",
    "The file type in the list_file. Currently support image and video.");
DEFINE_string(out_file, "",
    "If provided, store the detection results in the out_file.");
DEFINE_double(confidence_threshold, 0.2,
    "Only store detections with score higher than the threshold.");

int main(int argc, char** argv) {
  ::google::InitGoogleLogging(argv[0]);
  // Print output to stderr (while still logging)
  FLAGS_alsologtostderr = 1;

#ifndef GFLAGS_GFLAGS_H_
  namespace gflags = google;
#endif

  gflags::SetUsageMessage("Do detection using SSD mode.\n"
        "Usage:\n"
        "    ssd_detect [FLAGS] model_file weights_file list_file\n");
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  if (argc < 4) {
    gflags::ShowUsageWithFlagsRestrict(argv[0], "examples/ssd/ssd_detect");
    return 1;
  }

  const string& model_file = argv[1];
  const string& weights_file = argv[2];
  const string& mean_file = FLAGS_mean_file;
  const string& mean_value = FLAGS_mean_value;
  const string& file_type = FLAGS_file_type;
  const string& out_file = FLAGS_out_file;
  const float confidence_threshold = FLAGS_confidence_threshold;

  // Initialize the network.
  Detector detector(model_file, weights_file, mean_file, mean_value);

  // Set the output mode.
  std::streambuf* buf = std::cout.rdbuf();
  std::ofstream outfile;
  if (!out_file.empty()) {
    outfile.open(out_file.c_str());
    if (outfile.good()) {
      buf = outfile.rdbuf();
    }
  }
  std::ostream out(buf);

  // Process image one by one.
  std::ifstream infile(argv[3]);
  std::string file;
  int num_x=0;
  while (infile >> file) {
    if (file_type == "image") {
      char image[20];
      cv::Mat img = cv::imread(file, -1);
      CHECK(!img.empty()) << "Unable to decode image " << file;
 
      std::vector<vector<float> > detections = detector.Detect(img);
  //    std::vector<vector<float> > detections = detector.Detect(noparking);

      /* Print the detection results. */
      for (int i = 0; i < detections.size(); i=i+1) {
        const vector<float>& d = detections[i];
        // Detection format: [image_id, label, score, xmin, ymin, xmax, ymax].
        CHECK_EQ(d.size(), 7);
        const float score = d[2];
        if (score >= confidence_threshold) {
          out << file << " ";
          out << static_cast<int>(d[1]) << " ";
          out << score << " ";
          out << static_cast<int>(d[3] * img.cols) << " ";
          out << static_cast<int>(d[4] * img.rows) << " ";
          out << static_cast<int>(d[5] * img.cols) << " ";
          out << static_cast<int>(d[6] * img.rows) << std::endl;
	  float x=static_cast<int>(d[3] * img.cols);
          float y=static_cast<int>(d[4] * img.rows);
          float width=static_cast<int>(d[5] * img.cols)-x;
          float height=static_cast<int>(d[6] * img.rows)-y;
	  cv::Mat plate;
	  cv::Rect rect1(x,y,width,height);
            //cv::namedWindow("plate",1);
          img(rect1).copyTo(plate);
            //cv::imshow("original",img);  
           // cv::resize(plate,plate,cv::Size(136,36)); 
        //  cv::imshow("plate",plate); 
          //  cv::imshow("img",img);
          cv::waitKey(10);
          sprintf(image,"%s%d%s","image",num_x,".jpg");	
	  cv::imwrite(image,plate);
          num_x++;	
        }
      }
    } else if (file_type == "video") {
	

      
    //  char image_1[128];
    //  char image_2[128];
      static int count_noparking=0;
      //int count_original=0; 
      cv::VideoCapture cap(file);
      if (!cap.isOpened()) {
        LOG(FATAL) << "Failed to open video: " << file;
      }
      cv::Mat img;
    
      //cap>>img_1; 	
      int frame_count = 0;
      //static int frame_count1 = 0;
      while (true) {
        bool success = cap.read(img);
        if (!success) {
          LOG(INFO) << "Process " << frame_count << " frames from " << file;
          break;
        }
        CHECK(!img.empty()) << "Error when read frame";
          img_ori=img;
int count_original=0;
    //  namedWindow("yuanshi",3);
    //  imshow("yuanshi",img_ori);
   //   waitKey(1000);
        std::vector<vector<float> > detections = detector.Detect(img);
      //  cvtColor(img,img,CV_BGR2GRAY);
        LaneDetect detect(img);
      //  Mat img1;
       // cvtColor(img,img1,CV_BGR2GRAY);
        //LaneDetect detect(img1);
      
      //  cv::imshow("hahah",img);
	//cv::waitKey(1000);
     //   Mat img_noparking;
   //     Rect rect3(avr,160,480-avr,160);
     //   cout<<"avr="<<avr<<endl;
	//img(rect3).copyTo(img_noparking);
      //  namedWindow("noparking",3);
     //  imshow("noparking",noparking);
	//waitKey(10);
       cout<<noparking.cols<<"****"<<img.cols<<endl;
       cout<<noparking.rows<<"****"<<img.rows<<endl;
      // double ratio_x=noparking.cols/img.cols;
      // cout<<ratio_x<<endl;
       //float ratio_y=noparking.rows/img.rows;
      // cout<<ratio_x<<"*******"<<ratio_y<<endl;
       
        std::vector<vector<float> > detections1 = detector.Detect(noparking);
 /* Print the detection results. */
	//std::cout<<"size is "<<detections.size()<<std::endl;
        
        for (int i = 0; i < detections1.size(); i=i+1) {
     
        
        time_t rawtime;
	struct tm *info;
	char buffer[128];
	time(&rawtime);
	info =localtime(&rawtime);
	strftime(buffer,80,"%b%d_%H%M",info);
        string ss1=string(buffer);
	//ss1=buffer;
	//cout<<ss1<<endl;
	//std::cout<<ss1<<std::endl;      


          const vector<float>& d = detections1[i];
          // Detection format: [image_id, label, score, xmin, ymin, xmax, ymax].
          CHECK_EQ(d.size(), 7);
          const float score = d[2];
          int xz=0;
          if (score >= confidence_threshold) {
          //  cv::Mat img_1; 
	   // cap>>img_1;
            		 
            //std::cout<<"plate detected"<<std::endl;
            //out << "plate detected" <<" ";
	   // 
	    out << file << "_";
            out << std::setfill('0') << std::setw(6) << frame_count << " ";
            out << static_cast<int>(d[1]) << " ";
            out << score << " ";
            out << (static_cast<int>(d[3] * noparking.cols)+avr)*1920/480 << " ";
            out << (static_cast<int>(d[4] * noparking.rows)+160)*1080/320 << " ";
            out << (static_cast<int>(d[5] * noparking.cols)+avr)*1920/480 << " ";
            out << (static_cast<int>(d[6] * noparking.rows)+160)*1080/320 << std::endl;
            //cout<<noparking.cols<<" "<<noparking.rows<<endl;
	   // std::cout<<"a"<<std::endl;  little test when make caffe;
          //  float area;
	    //std::cout<<"area="<<(static_cast<int>(d[6] * img.rows)-static_cast<int>(d[4] * img.rows))*(static_cast<int>(d[5] * img.cols)*static_cast<int>(d[3] * img.cols))<<std::endl;
	    float x=(static_cast<int>(d[3] * noparking.cols)+avr)*1920/480;
            float y=(static_cast<int>(d[4] * noparking.rows)+160)*1080/320;
            float width=(static_cast<int>(d[5] * noparking.cols)+avr)*1920/480-x;
            float height=(static_cast<int>(d[6] * noparking.rows)+160)*1080/320-y;
            float xmax=x+width;
            float ymax=y+height;
            if(x<=0||y<=0||xmax>=1920||ymax>=1080){break;}
           // cv::rectangle(img,cvPoint(x,y),cvPoint(xmax,ymax),cvScalar(255,255,0),2);
           // namedWindow("originalimg");
           // imshow("originalimg",img);
            
            //float x=static_cast<int>(d[3] * img.cols);
           // float y=static_cast<int>(d[4] * img.rows);
           // float width=static_cast<int>(d[5] * img.cols)-x;
          //  float height=static_cast<int>(d[6] * img.rows)-y;
	    //std::cout<<"rect size is "<<x<<y<<width<<height<<std::endl;
	    
	    cv::Mat plate1;
	    cv::Rect rect1(x,y,width,height);
            //cv::namedWindow("plate",1);
            img(rect1).copyTo(plate1);
            //cv::imshow("original",img);  
           // cv::resize(plate,plate,cv::Size(136,36));
           // namedWindow("np",5); 
           // cv::imshow("np",plate1); 
          //  cv::imshow("img",img);
            cv::waitKey(10);
            string ss1_temp2=ss1;
            string count_no_parking="";
            int2str(count_noparking,count_no_parking);
            ss1+=count_no_parking;
            count_noparking++;
           // sprintf(image_1,"%s%d%s%s","image_noparking_",++count_noparking,ss1,".jpg");	
	   // cv::imwrite(image_1,plate);
             
        string s9;
        int2str(xz,s9);
        xz++;
          if(static_cast<int>(d[1])==2)
          { string ss2="plate_";
           string s1;
           int frame_count2=frame_count;
           int2str(frame_count2,s1);
           ss2=ss2+s1;
           ss2=ss2+"_";
           ss2=ss2+s9;
           	
            imwrite("/home/sdu/caffe-ssd/no_parking/"+ss2+".jpg",plate1);
             ss1=ss1_temp2;}
              //frame_count1++; 


          }
        }
        cout<<"&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&"<<endl;
        /* Print the detection results. */
	//std::cout<<"size is "<<detections.size()<<std::endl;
        for (int i = 0; i < detections.size(); i=i+1) {
                  time_t rawtime;
	struct tm *info;
	char buffer[128];
	time(&rawtime);
	info =localtime(&rawtime);
	strftime(buffer,80,"%b%d_%H%M",info);
        string ss1=string(buffer);
	//ss1=buffer;
	//cout<<ss1<<endl;
	//std::cout<<ss1<<std::endl;      
          const vector<float>& d = detections[i];
          // Detection format: [image_id, label, score, xmin, ymin, xmax, ymax].
          CHECK_EQ(d.size(), 7);
          const float score = d[2];
          if (score >= confidence_threshold) {
          //  cv::Mat img_1; 
	   // cap>>img_1;
            		 
            //std::cout<<"plate detected"<<std::endl;
            //out << "plate detected" <<" ";
	   // 
	    out << file << "_";
            out << std::setfill('0') << std::setw(6) << frame_count << " ";
            out << static_cast<int>(d[1]) << " ";
            out << score << " ";
            out << static_cast<int>(d[3] * img.cols) << " ";
            out << static_cast<int>(d[4] * img.rows) << " ";
            out << static_cast<int>(d[5] * img.cols) << " ";
            out << static_cast<int>(d[6] * img.rows) << std::endl;
	   // std::cout<<"a"<<std::endl;  little test when make caffe;
          //  float area;
	    //std::cout<<"area="<<(static_cast<int>(d[6] * img.rows)-static_cast<int>(d[4] * img.rows))*(static_cast<int>(d[5] * img.cols)*static_cast<int>(d[3] * img.cols))<<std::endl;
            cout<<"**********"<<static_cast<int>(d[1])<<endl;
	    float x=static_cast<int>(d[3] * img.cols);
            float y=static_cast<int>(d[4] * img.rows);
            float width=static_cast<int>(d[5] * img.cols)-x;
            float height=static_cast<int>(d[6] * img.rows)-y;
            float xmax=x+width;
            float ymax=y+height;
            if(static_cast<int>(d[6] * img.rows)>=img.rows||static_cast<int>(d[4] * img.rows)>=img.rows||static_cast<int>(d[5] * img.cols)>=img.cols||static_cast<int>(d[3] * img.cols)>=img.cols||static_cast<int>(d[3] * img.cols)<0||static_cast<int>(d[4] * img.rows)<0){break;}
           // cv::rectangle(img,cvPoint(x,y),cvPoint(xmax,ymax),cvScalar(255,255,0),2);
           // namedWindow("originalimg");
          //  imshow("originalimg",img);
            //float x=static_cast<int>(d[3] * img.cols);
           // float y=static_cast<int>(d[4] * img.rows);
           // float width=static_cast<int>(d[5] * img.cols)-x;
          //  float height=static_cast<int>(d[6] * img.rows)-y;
	    //std::cout<<"rect size is "<<x<<y<<width<<height<<std::endl;
	    string temp_ss1=ss1;
	    cv::Mat plate;
	    cv::Rect rect10(x,y,width,height);
            //cv::namedWindow("plate",1);
            img(rect10).copyTo(plate);
            //cv::imshow("original",img);  
           // cv::resize(plate,plate,cv::Size(136,36)); 
            cv::imshow("plate",plate); 
          //  cv::imshow("img",img);
            cv::waitKey(10);
            string count_all="";
            count_original++;
            int2str(count_original,count_all);
           // cout<<count_all<<endl;
            count_all="_"+count_all;
            
            ss1+=count_all;
            //count_all="";
           // cout<<"count_all1="<<count_all<<endl;
            
           // sprintf(image_2,"%s%d%s%s","image_",++count_original,string(ss1),".jpg");	
	  //  cv::imwrite(image_2,plate);
           string ss2="";
           if(static_cast<int>(d[1])==1)
           ss2="face_";
          
           else
           ss2="plate_";
        
           
           string ss3="";
           int frame_count2=frame_count;
           int2str(frame_count2,ss3);
           ss2=ss2+ss3;
           ss2=ss2+count_all;	
            imwrite("/home/sdu/caffe-ssd/original_rected/"+ss2+".jpg",plate);
         //  count_all="";
         //  cout<<"count_all2="<<count_all<<endl;
             ss1=temp_ss1;


          }

        }
	//int *stpr;
	 //sprintf(stpr,"frame_count%d");
         
        ++frame_count;

      }
      if (cap.isOpened()) {
        cap.release();
      }
    } else {
      LOG(FATAL) << "Unknown file_type: " << file_type;
    }
  }
  return 0;
}
#else
int main(int argc, char** argv) {
 // cout<<static_cast<int>(d[3]<<endl;
//  std::cout<<"hello"<<endl;
 // std::cout<<static_cast<int>(d[6] * img.rows) << std::endl;
  LOG(FATAL) << "This example requires OpenCV; compile with USE_OPENCV.";
}
#endif  // USE_OPENCV
