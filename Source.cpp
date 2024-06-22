#include <opencv2/opencv.hpp> // Include the OpenCV library
#include <iostream> // Include input-output stream library
#include <string> // Include string manipulation library

using namespace cv; // Use the cv namespace for OpenCV functions

// Function to draw connections between keypoints
void drawMatches(const Mat& templateImage, const std::vector<KeyPoint>& keypointsTemplate,
    const Mat& sceneImage, const std::vector<KeyPoint>& keypointsScene,
    const std::vector<DMatch>& matches, Mat& outputImage) {
    // Draw connections
    drawMatches(templateImage, keypointsTemplate, sceneImage, keypointsScene, matches, outputImage,
        Scalar::all(-1), Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
}

int main(int argc, char* argv[]) {
    // Check the number of input arguments
    if (argc != 5) {
        std::cerr << "Usage: " << argv[0] << " -sift <TemplateImagePath> <SceneImagePath> <OutputImagePath>" << std::endl;
        return -1;
    }

    // Parse command-line arguments
    std::string templateImagePath, sceneImagePath, outputImagePath;
    if (std::string(argv[1]) == "-sift") {
        templateImagePath = argv[2];
        sceneImagePath = argv[3];
        outputImagePath = argv[4];
    }
    else {
        std::cerr << "Invalid command format." << std::endl;
        return -1;
    }

    // Load template and scene images
    Mat templateImage = imread(templateImagePath);
    Mat sceneImage = imread(sceneImagePath);

    // Check if images are loaded successfully
    if (templateImage.empty() || sceneImage.empty()) {
        std::cerr << "Error: Unable to load images." << std::endl;
        return -1;
    }

    // Convert images to grayscale
    Mat templateGray, sceneGray;
    cvtColor(templateImage, templateGray, COLOR_BGR2GRAY);
    cvtColor(sceneImage, sceneGray, COLOR_BGR2GRAY);

    // Initialize SIFT detector
    Ptr<SIFT> sift = SIFT::create();

    // Detect keypoints and compute descriptors for template and scene images
    std::vector<KeyPoint> keypointsTemplate, keypointsScene;
    Mat descriptorsTemplate, descriptorsScene;
    sift->detectAndCompute(templateGray, noArray(), keypointsTemplate, descriptorsTemplate);
    sift->detectAndCompute(sceneGray, noArray(), keypointsScene, descriptorsScene);

    // Initialize Brute Force matcher
    BFMatcher matcher(NORM_L2);

    // Match descriptors
    std::vector<std::vector<DMatch>> knnMatches;
    matcher.knnMatch(descriptorsTemplate, descriptorsScene, knnMatches, 2);

    // Apply ratio test to filter good matches
    std::vector<DMatch> goodMatches;
    const float ratio_thresh = 0.75f;
    for (size_t i = 0; i < knnMatches.size(); i++) {
        if (knnMatches[i][0].distance < ratio_thresh * knnMatches[i][1].distance) {
            goodMatches.push_back(knnMatches[i][0]);
        }
    }

    // Draw connections
    Mat outputImage;
    drawMatches(templateImage, keypointsTemplate, sceneImage, keypointsScene, goodMatches, outputImage);

    //-- Localize the object
    std::vector<Point2f> obj;
    std::vector<Point2f> scene;
    for (int i = 0; i < goodMatches.size(); i++) {
        obj.push_back(keypointsTemplate[goodMatches[i].queryIdx].pt);
        scene.push_back(keypointsScene[goodMatches[i].trainIdx].pt);
    }

    // Find homography matrix
    Mat H = findHomography(obj, scene, RANSAC);

    //-- Get the corners from the template image
    std::vector<Point2f> objCorners(4);
    objCorners[0] = Point2f(0, 0);
    objCorners[1] = Point2f(templateImage.cols, 0);
    objCorners[2] = Point2f(templateImage.cols, templateImage.rows);
    objCorners[3] = Point2f(0, templateImage.rows);
    std::vector<Point2f> scene_corners(4);

    // Perspective transform
    perspectiveTransform(objCorners, scene_corners, H);

    //-- Draw boundaries
    line(outputImage, scene_corners[0] + Point2f(templateImage.cols, 0), scene_corners[1] + Point2f(templateImage.cols, 0), Scalar(0, 255, 0), 4);
    line(outputImage, scene_corners[1] + Point2f(templateImage.cols, 0), scene_corners[2] + Point2f(templateImage.cols, 0), Scalar(0, 255, 0), 4);
    line(outputImage, scene_corners[2] + Point2f(templateImage.cols, 0), scene_corners[3] + Point2f(templateImage.cols, 0), Scalar(0, 255, 0), 4);
    line(outputImage, scene_corners[3] + Point2f(templateImage.cols, 0), scene_corners[0] + Point2f(templateImage.cols, 0), Scalar(0, 255, 0), 4);

    // Save the result image
    imwrite(outputImagePath, outputImage);

    std::cout << "SIFT OKE AND SAVE : " << outputImagePath << std::endl;

    return 0;
}
