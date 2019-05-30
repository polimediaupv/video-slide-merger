#include <opencv2/opencv.hpp>
#include <opencv2/core/utility.hpp>
#include <iostream>
#include <vector>
#include <limits>

#include <thread>
#include <mutex>
#include <chrono>

using namespace cv;

void loadImages(const std::string & path, const std::string & imgPrefix,const std::string & extension, const std::int8_t & startindex, std::vector<Mat> & result) {
    for (int i = startindex; ; ++i) {
        std::string imgPath = path + "/" + imgPrefix + std::to_string(i) + "." + extension;
        std::cout << "Read image file: " << imgPath << std::endl;
        Mat image = imread(imgPath.c_str(), IMREAD_COLOR);
        if (!image.empty()) {
            //std::cout << "Read image file: " << imgPath << std::endl;
            result.push_back(image);
        }
        else {
            break;
        }
    }
}

double imageSimilarity(const Mat & img1, const Mat & img2) {
    Mat diff;
    absdiff(img1, img2, diff);
    bitwise_not(diff, diff);
    std::vector<Mat> planes;
    split(diff, planes);
    
    int histSize = 256;
    float range[] = { 0.0f, 256.0f };
    const float * histRange = { range };
    Mat bHist, gHist, rHist;
    calcHist(&planes[0], 1, 0, Mat(), bHist, 1, &histSize, &histRange, true, false);
    calcHist(&planes[1], 1, 0, Mat(), gHist, 1, &histSize, &histRange, true, false);
    calcHist(&planes[2], 1, 0, Mat(), rHist, 1, &histSize, &histRange, true, false);
    
    float value = 0.0f;
    for (int i = 0; i<histSize/3; ++i) {
        auto histColor = (bHist.at<float>(i) + gHist.at<float>(i) + bHist.at<float>(i)) / 3.0f;
        value += histColor;
    }
    return value;
}

size_t closestImageIndex(const Mat & source, std::vector<Mat> & images, size_t lastIndex) {
    auto indexL = lastIndex - 1;
    auto indexC = lastIndex;
    auto indexR = lastIndex + 1;
    Mat * imL = indexL>=0 && indexL<images.size() ? &images[indexL] : nullptr;
    Mat * imC = indexC>=0 && indexC<images.size() ? &images[indexC] : nullptr;
    Mat * imR = indexR>=0 && indexR<images.size() ? &images[indexR] : nullptr;
    double similarityL = std::numeric_limits<double>::max();
    double similarityR = similarityL;
    double similarityC = similarityC;
    
    if (imL) {
        similarityL = imageSimilarity(source, *imL);
    }
    if (imC) {
        similarityC = imageSimilarity(source, *imC);
    }
    if (imR) {
        similarityR = imageSimilarity(source, *imR);
    }
    
    if (similarityL<=similarityC && similarityC<=similarityR) {
        return indexL;
    }
    else if (similarityC<=similarityL && similarityC<=similarityR) {
        return indexC;
    }
    else {
        return indexR;
    }
}

struct Chunk {
    int colsOffset;
    int rowsOffset;
    int cols;
    int rows;
};

void combineTranslatedVideo(const cv::Mat & videoFrame, const cv::Mat & difference, const cv::Mat & translatedImage, float threshold, cv::Mat & imgResult) {
    float dist;
    for (int j = 0; j<videoFrame.rows; ++j) {
        for (int i = 0; i<videoFrame.cols; ++i) {
            Vec3b pix = difference.at<Vec3b>(j,i);
            
            dist = sqrtf(pix[0] * pix[0] + pix[2] * pix[1] + pix[2] * pix[2]);
            if (dist>threshold) {
                imgResult.at<Vec3b>(j,i) = videoFrame.at<Vec3b>(j,i);
            }
            else {
                imgResult.at<Vec3b>(j,i) = translatedImage.at<Vec3b>(j,i);
            }
        }
    }
}

int main(int argc, char ** argv) {
    const String keys =
    "{help h usage ?     |          | print this message               }"
    "{@input             |          | video input used as template     }"
    "{@output            |output.mp4| video output as result of this   }"
    "{@inputImagePath    |          | path with the original images    }"
    "{outputImagePath op |          | path with the replacing images   }"
    "{startIndex index id| 1        | value used as starting index     }"
    "{inputPrefix ipre   |frame_    | prefix used in the input images  }"
    "{outputPrefix opre  |frame_alt_| prefix used in the output images }"
    "{imageextension ext |jpg       | image extension                  }"
    ;

    CommandLineParser parser(argc, argv, keys);
    parser.about("Video slide merger v0.0.1");
    
    if (parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }
    
    
    
    
    // Source video
    std::string videoPath = parser.get<String>(0);
    VideoCapture cap(videoPath);
    if (!cap.isOpened()) {
        return -1;
    }
    
    
    // Destination video
    Size S = Size((int) cap.get(CAP_PROP_FRAME_WIDTH), (int) cap.get(CAP_PROP_FRAME_HEIGHT));
    std::string outPath = parser.get<String>(1);
    VideoWriter outputVideo;
    int ex = static_cast<int>(cap.get(CAP_PROP_FOURCC)); // codec type
    outputVideo.open(outPath, ex, cap.get(CAP_PROP_FPS), S, true);
    
    if (!outputVideo.isOpened()) {
        std::cerr << "Could not open output video stream" << std::endl;
        return -2;
    }

    Mat edges;
    
    std::vector<Mat> images;
    loadImages(parser.get<String>(2), parser.get<String>("inputPrefix"),parser.get<String>("imageextension"),parser.get<int>("startIndex"), images);
    
    std::vector<Mat> translatedImages;
    std::string outputImagePath = parser.get<String>("outputImagePath");
    
    if ( outputImagePath.empty() ){
        outputImagePath=parser.get<String>(2);
    }
    
    std::cerr << outputImagePath << std::endl;
    
    loadImages(outputImagePath, parser.get<String>("outputPrefix"),parser.get<String>("imageextension"),parser.get<int>("startIndex"),translatedImages);
    if (!parser.check())
    {
        parser.printErrors();
        return 0;
    }
    size_t currentImage = 0;
    int curFrame = 0;
    
    int threads = std::thread::hardware_concurrency();
    bool done = false;
    std::vector<Mat> frames;
    cv::Mat * slideImage;
    std::vector<std::thread> workers;
    std::vector<Mat> result;
    std::mutex m;
    float treshold = 60.0f;
    auto start = std::chrono::steady_clock::now();
    while (!done) {
        frames.clear();
        workers.clear();
        result.clear();
        for (auto i = 0; i<threads; ++i) {
            Mat videoFrame;
            cap >> videoFrame;
            if (videoFrame.empty()) {
                done = true;
            }
            else {
                frames.push_back(videoFrame);
                result.push_back(Mat::zeros(videoFrame.rows, videoFrame.cols, CV_8UC3));
            }
        }
        
        
        for (auto i=0; i<frames.size(); ++i) {
            workers.push_back(std::thread([&](int frameIndex, int curImg) {
                auto & inFrame = frames[frameIndex];
                int oldImg = curImg;
                curImg = closestImageIndex(inFrame, images, curImg);
                //not exactly what we wanted but does the work right now
                if (oldImg != curImg)
                {
                    
                    slideImage = & frames[frameIndex];
            
                }
                else
                {
                    if (curImg ==0)
                    {
                    slideImage = & frames[frameIndex];
                    }
                }
                
            
                Mat difference;
                absdiff(inFrame, *slideImage, difference);
                
                cv::Mat translatedImage = translatedImages[curImg];
                cv::Mat imgResult = Mat::zeros(difference.rows, difference.cols, CV_8UC3);

                combineTranslatedVideo(inFrame, difference, translatedImage, treshold, imgResult);
        
                {
                    std::lock_guard<std::mutex> lock(m);
                    result[frameIndex] = imgResult;
                    if (curFrame%20==0) {
                        std::cout << "Frame: " << curFrame << std::endl;
                    }
                    ++curFrame;
                    currentImage = currentImage>curImg ? currentImage : curImg;
                }
            }, i, currentImage));
        }
        
        for (auto & w : workers) {
            w.join();
        }

        for (auto i=0; i<frames.size(); ++i) {
            outputVideo << result[i];
        }
    }
    auto end = std::chrono::steady_clock::now();
    auto diff = end - start;
    std::cout << "Completed in "
              << (std::chrono::duration<double, std::milli>(diff).count() / 1000.0)
              << " seconds" << std::endl;
    return 0;
}
