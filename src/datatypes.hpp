#include <opencv4/opencv2/opencv.hpp>
#include <opencv4/opencv2/highgui.hpp>
#include <vector>
#include <string>

struct directory_datatype
{
    std::string read_directory_path_1 = "";
    std::string read_directory_path_2 = "";
    std::string save_directory_path = "";
    std::string detections_directory_path = "";
    std::string matches_directory_path = "";
};

struct im_tools
{
    cv::Ptr<cv::Feature2D> detector = cv::ORB::create(500);
    cv::Ptr<cv::StereoBM> stereo = cv::StereoBM::create(16, 21);
    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::BRUTEFORCE_HAMMING);
};

struct point3D_datatype
{
    cv::Point3f point3D;         // 3D coordinates
    cv::Vec3f viewing_direction; // Viewing direction
    int best_descriptor_idx;      // Index of best ORB descriptor
    float dmin, dmax;            // Visibility depth range
};

struct match_datatype
{
    cv::Mat img1 = {};
    cv::Mat img2 = {};
    std::vector<cv::KeyPoint> keypoints1 = {};
    std::vector<cv::KeyPoint> keypoints2 = {};
    std::vector<std::vector<cv::DMatch>> matches = {};
    std::vector<cv::Point2f> points2f_1 = {};
    std::vector<cv::Point2f> points2f_2 = {};
    cv::Mat descriptors1 = {};
    cv::Mat descriptors2 = {};

    std::vector<point3D_datatype> points3D; // Store 3D point structures

    void convert_keypoints_to_points()
    {
        cv::KeyPoint::convert(this->keypoints1, this->points2f_1);
        cv::KeyPoint::convert(this->keypoints2, this->points2f_2);
    }
};

struct feature_datatype
{
    cv::Mat img = {};
    std::vector<cv::KeyPoint> keypoints = {};
    cv::Mat descriptors = {};
};

struct Keyframe {
    cv::Mat rotation;
    cv::Mat translation;
    std::vector<cv::KeyPoint> keypoints;
    std::vector<cv::Point2f> points2f;
    cv::Mat descriptors;
    std::vector<point3D_datatype> points3D; 
};