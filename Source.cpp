#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using namespace std;
using namespace cv;

//1
void load_image(const string& input_path, Mat& image) {
    // Load the input image 
    image = imread(input_path);
    if (image.empty()) {
        cerr << "Error: Unable to load the image from the specified path." << endl;
        exit(1);
    }
}

void show_and_save_image(const Mat& image, const string& output_path) {
    // Show the result image
    imshow("Result Image", image);
    waitKey(0);

    // Save the result image
    imwrite(output_path, image);
    cout << "Result image saved successfully at " << output_path << endl;
}

int convert_to_gray(const cv::Mat& srcImage, cv::Mat& dstImage) {
    // Kiểm tra ảnh đầu vào
    if (srcImage.empty())
        return 0;

    // Lấy kích thước của ảnh
    int height = srcImage.rows;
    int width = srcImage.cols;

    // Tạo ảnh kết quả với định dạng grayscale
    dstImage = cv::Mat(height, width, CV_8UC1);

    // Lặp qua từng pixel của ảnh đầu vào
    for (int y = 0; y < height; ++y) {
        const uchar* pSrcRow = srcImage.ptr<uchar>(y);
        uchar* pDstRow = dstImage.ptr<uchar>(y);

        for (int x = 0; x < width; ++x) {
            // Tính giá trị trung bình của các kênh màu (B, G, R)
            // và gán vào kênh màu duy nhất của ảnh grayscale
            pDstRow[x] = (uchar)((pSrcRow[3 * x] + pSrcRow[3 * x + 1] + pSrcRow[3 * x + 2]) / 3);
        }
    }

    return 1;
}
// Thay đổi độ sáng của ảnh
int ChangeBrightness(const cv::Mat& srcImage, cv::Mat& dstImage, int brightness) {
    // Kiểm tra ảnh đầu vào không rỗng
    if (srcImage.empty())
        return 0;

    // Thực hiện thay đổi độ sáng
    dstImage = srcImage + brightness;

    return 1; // Trả về 1 nếu thay đổi thành công
}




// thay đổi độc tương phản 
int ChangeContrast(const Mat& srcImage, Mat& dstImage, float contrast) {
    // Kiểm tra ảnh đầu vào không rỗng
    if (srcImage.empty())
        return 0;

    // Thực hiện thay đổi độ tương phản
    dstImage = srcImage * contrast;

    return 1; // Trả về 1 nếu thay đổi thành công
}



// Hàm avg_filter
int avg_filter(const Mat& inputImage, Mat& outputImage, int kernelSize) {
    // Kiểm tra ảnh đầu vào không rỗng
    if (inputImage.empty())
        return 0; // Trả về 0 nếu không thành công

    // Kiểm tra kích thước kernel hợp lệ
    if (kernelSize <= 0 || kernelSize % 2 == 0)
        return -1; // Trả về -1 nếu kích thước kernel không hợp lệ

    // Tạo ma trận lưu trữ ảnh kết quả
    outputImage = Mat::zeros(inputImage.size(), inputImage.type());

    // Lặp qua từng kênh màu của ảnh
    for (int channel = 0; channel < inputImage.channels(); ++channel) {
        // Lặp qua từng pixel của ảnh đầu vào
        for (int y = 0; y < inputImage.rows; ++y) {
            for (int x = 0; x < inputImage.cols; ++x) {
                int sum = 0;
                int count = 0;

                // Tính tổng giá trị các pixel trong kernel
                for (int ky = -kernelSize / 2; ky <= kernelSize / 2; ++ky) {
                    for (int kx = -kernelSize / 2; kx <= kernelSize / 2; ++kx) {
                        int ny = y + ky;
                        int nx = x + kx;
                        if (ny >= 0 && ny < inputImage.rows && nx >= 0 && nx < inputImage.cols) {
                            sum += inputImage.at<Vec3b>(ny, nx)[channel];
                            count++;
                        }
                    }
                }

                // Tính giá trị trung bình và gán vào pixel tương ứng trong ảnh kết quả
                outputImage.at<Vec3b>(y, x)[channel] = saturate_cast<uchar>(sum / count);
            }
        }
    }

    return 1; // Trả về 1 nếu thành công
}



int median_filter(const Mat& inputImage, Mat& outputImage, int kernelSize) {
    // Kiểm tra ảnh đầu vào không rỗng
    if (inputImage.empty())
        return 0; // Trả về 0 nếu không thành công

    // Kiểm tra kích thước kernel hợp lệ
    if (kernelSize <= 0 || kernelSize % 2 == 0)
        return -1; // Trả về -1 nếu kích thước kernel không hợp lệ

    // Tạo ma trận lưu trữ ảnh kết quả
    outputImage = Mat::zeros(inputImage.size(), inputImage.type());

    // Lặp qua từng kênh màu của ảnh
    for (int channel = 0; channel < inputImage.channels(); ++channel) {
        // Lặp qua từng pixel của ảnh đầu vào
        for (int y = 0; y < inputImage.rows; ++y) {
            for (int x = 0; x < inputImage.cols; ++x) {
                vector<uchar> values;

                // Tính tổng giá trị các pixel trong kernel
                for (int ky = -kernelSize / 2; ky <= kernelSize / 2; ++ky) {
                    for (int kx = -kernelSize / 2; kx <= kernelSize / 2; ++kx) {
                        int ny = y + ky;
                        int nx = x + kx;
                        if (ny >= 0 && ny < inputImage.rows && nx >= 0 && nx < inputImage.cols) {
                            values.push_back(inputImage.at<Vec3b>(ny, nx)[channel]);
                        }
                    }
                }

                // Sắp xếp giá trị và lấy giá trị trung vị
                sort(values.begin(), values.end());
                uchar medianValue = values[values.size() / 2];

                // Gán giá trị trung vị vào pixel tương ứng trong ảnh kết quả
                outputImage.at<Vec3b>(y, x)[channel] = medianValue;
            }
        }
    }

    return 1; // Trả về 1 nếu thành công
}




Mat getGaussianKernel_byhand(int ksize, double sigma, int ktype) {
    // Tạo ma trận Gaussian kernel có kích thước ksize x 1
    Mat kernel = Mat::zeros(ksize, 1, ktype);

    // Tính toán trung tâm của kernel
    int center = ksize / 2;

    // Tính toán hệ số chuẩn hóa
    double sigma2 = sigma * sigma;
    double sum = 0.0;

    // Tính toán giá trị của kernel
    for (int i = 0; i < ksize; ++i) {
        double x = i - center;
        double g = exp(-(x * x) / (2 * sigma2));
        kernel.at<float>(i) = static_cast<float>(g);
        sum += g;
    }

    // Chuẩn hóa kernel
    kernel /= sum;

    return kernel;
}

int apply_gaussian_filter(const Mat& inputImage, Mat& outputImage, int kernelSize) {
    // Kiểm tra kích thước kernel
    if (kernelSize <= 0 || kernelSize % 2 == 0) {
        // Trả về mã lỗi nếu kích thước kernel không hợp lệ
        return -1;
    }

    // Tạo kernel Gaussian
    Mat gaussianKernelX = getGaussianKernel_byhand(kernelSize, -1, CV_32F);
    Mat gaussianKernelY = getGaussianKernel_byhand(kernelSize, -1, CV_32F);
    Mat gaussianKernel2D = gaussianKernelX * gaussianKernelY.t();

    // Sao chép ảnh đầu vào vào ảnh đầu ra
    outputImage = inputImage.clone();

    // Biên của kernel
    int border = kernelSize / 2;

    // Áp dụng bộ lọc Gaussian
    for (int y = border; y < inputImage.rows - border; ++y) {
        for (int x = border; x < inputImage.cols - border; ++x) {
            Vec3f sum(0, 0, 0);
            for (int ky = -border; ky <= border; ++ky) {
                for (int kx = -border; kx <= border; ++kx) {
                    int nx = x + kx;
                    int ny = y + ky;
                    Vec3f pixelValue = inputImage.at<Vec3b>(ny, nx);
                    float gaussianWeight = gaussianKernel2D.at<float>(ky + border, kx + border);
                    sum += pixelValue * gaussianWeight;
                }
            }
            outputImage.at<Vec3b>(y, x) = sum;
        }
    }

    // Trả về 1 để chỉ ra thành công
    return 1;
}



int detect_edges_sobel(const Mat& inputImage, Mat& outputImage) {
    // Check if the input image is grayscale
    Mat image_in = inputImage.clone();
    if (inputImage.channels() != 1) {
        convert_to_gray(inputImage, image_in);
    }

    // Define kernel size (commonly 3x3)
    int kernelSize = 3;
    int border = kernelSize / 2;

    // Define Sobel kernels
    int sobelKernelX[3][3] = { {-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1} };
    int sobelKernelY[3][3] = { {-1, -2, -1}, {0, 0, 0}, {1, 2, 1} };

    // Allocate memory for intermediate results
    Mat gradX(image_in.size(), CV_16SC1);
    Mat gradY(image_in.size(), CV_16SC1);

    // Apply Sobel filter
    for (int y = border; y < image_in.rows - border; ++y) {
        for (int x = border; x < image_in.cols - border; ++x) {
            int sumX = 0, sumY = 0;
            for (int ky = -border; ky <= border; ++ky) {
                for (int kx = -border; kx <= border; ++kx) {
                    int nx = x + kx;
                    int ny = y + ky;
                    int pixelValue = image_in.at<uchar>(ny, nx);
                    sumX += pixelValue * sobelKernelX[ky + border][kx + border];
                    sumY += pixelValue * sobelKernelY[ky + border][kx + border];
                }
            }
            gradX.at<short>(y, x) = sumX;
            gradY.at<short>(y, x) = sumY;
        }
    }

    // Convert gradients to absolute values
    convertScaleAbs(gradX, gradX);
    convertScaleAbs(gradY, gradY);

    // Combine gradients to compute edge magnitude
    outputImage.create(image_in.size(), CV_8UC1);
    for (int y = 0; y < image_in.rows; ++y) {
        for (int x = 0; x < image_in.cols; ++x) {
            outputImage.at<uchar>(y, x) = saturate_cast<uchar>(sqrt(pow(gradX.at<uchar>(y, x), 2) + pow(gradY.at<uchar>(y, x), 2)));
        }
    }

    return 1; // Return success
}



int detect_edges_lap(const Mat& inputImage, Mat& outputImage) {
    // Check if the input image is grayscale
    Mat image_in;
    convert_to_gray(inputImage, image_in);

    // Define kernel size (commonly 3x3)
    int kernelSize = 3;
    int border = kernelSize / 2;

    // Define Laplacian kernel
    int laplacianKernel[3][3] = { {0, 1, 0}, {1, -4, 1}, {0, 1, 0} };

    // Allocate memory for the output image
    outputImage.create(image_in.size(), CV_8UC1);

    // Apply Laplacian filter
    for (int y = border; y < image_in.rows - border; ++y) {
        for (int x = border; x < image_in.cols - border; ++x) {
            int sum = 0;
            for (int ky = -border; ky <= border; ++ky) {
                for (int kx = -border; kx <= border; ++kx) {
                    int nx = x + kx;
                    int ny = y + ky;
                    int pixelValue = image_in.at<uchar>(ny, nx);
                    sum += pixelValue * laplacianKernel[ky + border][kx + border];
                }
            }
            // Saturate the value to fit the output format
            outputImage.at<uchar>(y, x) = saturate_cast<uchar>(sum);
        }
    }

    // Trả về 1 để chỉ ra thành công
    return 1;
}






int main(int argc, char* argv[]) {
    // Kiểm tra số lượng đối số dòng lệnh
    if (argc < 4 || argc > 6) {
        cerr << "Usage: " << argv[0] << " <filter_type> <InputFilePath> <OutputFilePath> <Parameter>" << endl;
        return 1;
    }

    // Lấy thông tin từ dòng lệnh
    string filterType = argv[1];
    string inputPath = argv[2];
    string outputPath = argv[3];

    // Xử lý tùy theo loại bộ lọc
    Mat inputImage, outputImage;
    int result = -1;

    if (filterType == "-rgb2gray") {
        // Chuyển đổi ảnh màu sang ảnh xám
        load_image(inputPath, inputImage);
        result = convert_to_gray(inputImage, outputImage);
    }
    else if (filterType == "-brightness") {
        // Thay đổi độ sáng của ảnh
        float brightnessFactor = atof(argv[4]);
        load_image(inputPath, inputImage);
        result = ChangeBrightness(inputImage, outputImage, brightnessFactor);
    }
    else if (filterType == "-contrast") {
        // Thay đổi độ tương phản của ảnh
        float contrastFactor = atof(argv[4]);
        load_image(inputPath, inputImage);
        result = ChangeContrast(inputImage, outputImage, contrastFactor);
    }
    else if (filterType == "-avg") {
        // Áp dụng bộ lọc trung bình
        int kernelSize = atoi(argv[4]);
        load_image(inputPath, inputImage);
        result = avg_filter(inputImage, outputImage, kernelSize);
    }
    else if (filterType == "-med") {
        // Áp dụng bộ lọc trung vị
        int kernelSize = atoi(argv[4]);
        load_image(inputPath, inputImage);
        result = median_filter(inputImage, outputImage, kernelSize);
    }
    else if (filterType == "-gau") {
        // Áp dụng bộ lọc Gauss
        int kernelSize = atoi(argv[4]);
        load_image(inputPath, inputImage);
        result = apply_gaussian_filter(inputImage, outputImage, kernelSize);
    }
    else if (filterType == "-sobel") {
        // Phát hiện biên cạnh bằng bộ lọc Sobel
        load_image(inputPath, inputImage);
        result = detect_edges_sobel(inputImage, outputImage);
    }
    else if (filterType == "-laplace") {
        // Phát hiện biên cạnh bằng phép Laplace
        load_image(inputPath, inputImage);
        result = detect_edges_lap(inputImage, outputImage);
    }
    else {
        cerr << "Error: Invalid filter type. Please use one of the following: -rgb2gray, -brightness, -contrast, -avg, -med, -gau, -sobel, -laplace." << endl;
        return 1;
    }

    // Kiểm tra kết quả và lưu ảnh
    if (result != 1) {
        cerr << "Error: Unable to apply filter." << endl;
        return -1;
    }

    show_and_save_image(outputImage, outputPath);

    return 0;
}