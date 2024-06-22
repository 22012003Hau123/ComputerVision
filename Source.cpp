#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

// Load the image
Mat loadImage(const string& imagePath) {
    Mat image = imread(imagePath, IMREAD_COLOR);
    if (image.empty()) {
        cout << "Error loading image" << endl;
    }
    return image;
}

int convert_to_gray(const cv::Mat& srcImage, cv::Mat& dstImage) {
    
    if (srcImage.empty())
        return 0;

    int height = srcImage.rows;
    int width = srcImage.cols;

    dstImage = cv::Mat(height, width, CV_8UC1);

    for (int y = 0; y < height; ++y) {
        const uchar* pSrcRow = srcImage.ptr<uchar>(y);
        uchar* pDstRow = dstImage.ptr<uchar>(y);
        for (int x = 0; x < width; ++x) {
            pDstRow[x] = (uchar)((pSrcRow[3 * x] + pSrcRow[3 * x + 1] + pSrcRow[3 * x + 2]) / 3);
        }
    }

    return 1;
}

Mat convertToGray(const Mat& colorImage) {
    Mat grayImage;
    convert_to_gray(colorImage, grayImage);
    return grayImage;
}

Mat getGaussianKernel1(int ksize, double sigma, int type) {
    Mat kernel(ksize, ksize, type);
    double sigmaX = sigma > 0 ? sigma : 0.3 * ((ksize - 1) * 0.5 - 1) + 0.8;
    double sum = 0.0;

    for (int i = 0; i < ksize; i++) {
        for (int j = 0; j < ksize; j++) {
            double x = i - ksize * 0.5;
            double y = j - ksize * 0.5;
            double g = exp(-(x * x + y * y) / (2 * sigmaX * sigmaX));
            kernel.at<double>(i, j) = g;
            sum += g;
        }
    }

    kernel /= sum;

    return kernel;
}
void GaussianBlurr(InputArray src, OutputArray dst, Size ksize,double sigmaX, double sigmaY, int borderType) {

    if (ksize.width % 2 == 0 || ksize.height % 2 == 0) {
        CV_Error(Error::StsBadArg, "ksize.width and ksize.height must be odd!");
    }

    Mat kernel = getGaussianKernel1(ksize.width, sigmaX, CV_64F);
    kernel = kernel.t() * kernel;
    int depth = src.depth();

    filter2D(src, dst, depth, kernel, Point(-1, -1), 0, borderType);
}



void Harris(const Mat& src_gray, Mat& dst, double k = 0.04) {
    if (src_gray.empty() || src_gray.depth() != CV_8U || src_gray.channels() != 1) {
        std::cerr << "Invalid input image: must be a single-channel 8-bit grayscale image." << std::endl;
        return;
    }

    // Calculate image derivatives
    Mat abs_grad_x, abs_grad_y;
    Sobel(src_gray, abs_grad_x, CV_32FC1, 1, 0, 3, 1, 0, BORDER_REPLICATE); // Use BORDER_REPLICATE for consistency
    Sobel(src_gray, abs_grad_y, CV_32FC1, 0, 1, 3, 1, 0, BORDER_REPLICATE);

    Mat x2_derivative, y2_derivative, xy_derivative;
    pow(abs_grad_x, 2.0, x2_derivative);
    pow(abs_grad_y, 2.0, y2_derivative);
    multiply(abs_grad_x, abs_grad_y, xy_derivative);

    Mat x2g_derivative, y2g_derivative, xyg_derivative;
    GaussianBlurr(x2_derivative, x2g_derivative, Size(7, 7), 2.0, 0.0, BORDER_REPLICATE);
    GaussianBlurr(y2_derivative, y2g_derivative, Size(7, 7), 0.0, 2.0, BORDER_REPLICATE);
    GaussianBlurr(xy_derivative, xyg_derivative, Size(7, 7), 2.0, 2.0, BORDER_REPLICATE);

    Mat x2y2, xy ,mtrace;
    multiply(x2g_derivative, y2g_derivative, x2y2);
    multiply(xyg_derivative, xyg_derivative, xy);
    pow(x2g_derivative + y2g_derivative, 2.0, mtrace);
    dst = x2y2 - xy - k * mtrace;
}


Mat detectHarris(const Mat& grayImage) {
    Mat output;
    Harris(grayImage, output, 0.04);
    return output;
}
void convertScaleAbsManual(const Mat& src, Mat& dst) {
    dst.create(src.size(), CV_8UC1); 

    for (int i = 0; i < src.rows; ++i) {
        const float* srcRow = src.ptr<float>(i); 
        uchar* dstRow = dst.ptr<uchar>(i); 

        for (int j = 0; j < src.cols; ++j) {
            dstRow[j] = saturate_cast<uchar>(abs(srcRow[j])); 
        }
    }
}
// ve hinh tron
void draw_cirle(Mat& image, const Mat& cornerMap) {
    Mat outputNorm, outputNormScaled;
    normalize(cornerMap, outputNorm, 0, 255, NORM_MINMAX, CV_32FC1, Mat());
    convertScaleAbsManual(outputNorm, outputNormScaled);

    for (int j = 0; j < outputNorm.rows; j++) {
        for (int i = 0; i < outputNorm.cols; i++) {
            if ((int)outputNorm.at<float>(j, i) > 100) {
                circle(image, Point(i, j), 4, Scalar(0, 0, 255), 1, 8, 0);
            }
        }
    }
}

// out anh

void harrisCornerDetector(const string& imagePath) {
    Mat image = loadImage(imagePath);
    if (!image.empty()) {
        imshow("Original image", image);
        Mat grayImage = convertToGray(image);
        Mat cornerMap = detectHarris(grayImage);
    }
}
void displayAndSaveResult(const Mat& image, const std::string& outputImagePath) {
    imshow("Output Harris", image);
    imwrite(outputImagePath, image);
    waitKey(0);
}

int main(int argc, char* argv[]) {
    if (argc != 4 || std::string(argv[1]) != "-harris") {
        cout << "Usage: <Executable file> -harris <InputFilePath> <OutputFilePath>" << endl;
        return 1;
    }

    string inputImagePath = argv[2];
    string outputImagePath = argv[3];

    Mat image = loadImage(inputImagePath);
    if (image.empty()) {
        cout << "Error loading image" << endl;
        return 1;
    }

    Mat grayImage = convertToGray(image);

    Mat cornerMap = detectHarris(grayImage);

    draw_cirle(image, cornerMap);

    displayAndSaveResult(image, outputImagePath);
    return 0;
}
