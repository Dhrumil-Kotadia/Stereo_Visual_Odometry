#include <opencv4/opencv2/opencv.hpp>
#include <opencv4/opencv2/highgui.hpp>
#include <vector>
#include <Eigen/Dense>
#include "datatypes.hpp"

void get_disparity_map(cv::Mat img1, cv::Mat img2, cv::Mat &disparity_map, im_tools tools)
{
    tools.stereo->compute(img1, img2, disparity_map);
}

void update_keyframe(Keyframe &keyframe, cv::Mat rotation, cv::Mat translation, std::vector<cv::KeyPoint> keypoints, std::vector<cv::Point2f> points2f, cv::Mat descriptors, std::vector<point3D_datatype> map_points) 
{
    keyframe.rotation = rotation.clone();
    keyframe.translation = translation.clone();
    keyframe.keypoints = keypoints;
    keyframe.points2f = points2f;
    keyframe.descriptors = descriptors;
    keyframe.points3D = map_points;
}

cv::Mat bgr_to_gray(const cv::Mat &img) 
{
    cv::Mat gray;
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    return gray;
}

cv::Mat resize_image(const cv::Mat &img, int width, int height) 
{
    cv::Mat resized;
    cv::resize(img, resized, cv::Size(width, height));
    return resized;
}

feature_datatype extract_features(const cv::Mat img, cv::Ptr<cv::Feature2D> detector) 
{
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    detector->detectAndCompute(img, cv::noArray(), keypoints, descriptors);
    // std::cout << "Features Extracted" << std::endl;
    feature_datatype features;
    features.img = img;
    features.keypoints = keypoints;
    features.descriptors = descriptors;

    return features;
}

cv::Mat draw_keypoints(match_datatype match_data) 
{
    cv::Mat out;
    cv::drawKeypoints(match_data.img1, match_data.keypoints1, out);

    return out;
}

cv::Mat draw_valid_detections(match_datatype match_data, cv::Mat img)
{
    cv::Mat out;
    out = img.clone();
    for (int i = 0; i < match_data.points2f_2.size(); i++)
    {
        cv::circle(out, match_data.points2f_2[i], 3, cv::Scalar(0, 255, 0), -1);
    }
    return out;

}

match_datatype match_features(const cv::Mat img1, const cv::Mat img2, im_tools tools) 
{
    feature_datatype feature_data_1 = extract_features(img1, tools.detector);
    feature_datatype feature_data_2 = extract_features(img2, tools.detector);
    std::vector<std::vector<cv::DMatch>> matches;
    tools.matcher->knnMatch(feature_data_1.descriptors, feature_data_2.descriptors, matches, 3000);

    match_datatype result;
    result.img1 = img1;
    result.img2 = img2;
    result.keypoints1 = feature_data_1.keypoints;
    result.keypoints2 = feature_data_2.keypoints;
    result.matches = matches;
    result.descriptors1 = feature_data_1.descriptors;
    result.descriptors2 = feature_data_2.descriptors;
    result.convert_keypoints_to_points();

    // std::cout << "num_matches_returned: " << result.matches.size() << std::endl;

    return result;
}

cv::Mat draw_matches(match_datatype match_data) 
{
    cv::Mat out;
    std::vector<cv::DMatch> inlier_matches;

    // Create matches for inliers
    for (int i = 0; i < match_data.points2f_1.size(); i++) 
    {
        // Manually find the index of the keypoint that matches the inlier point
        int idx1 = -1;
        int idx2 = -1;
        
        for (int j = 0; j < match_data.keypoints1.size(); j++) 
        {
            if (match_data.keypoints1[j].pt == match_data.points2f_1[i]) 
            {
                idx1 = j;
                break;
            }
        }

        for (int j = 0; j < match_data.keypoints2.size(); j++) 
        {
            if (match_data.keypoints2[j].pt == match_data.points2f_2[i]) 
            {
                idx2 = j;
                break;
            }
        }

        if (idx1 != -1 && idx2 != -1) 
        {
            inlier_matches.push_back(cv::DMatch(idx1, idx2, 0));
        }
    }

    // Now draw the matches with only inliers
    cv::drawMatches(match_data.img1, match_data.keypoints1, match_data.img2, match_data.keypoints2, inlier_matches, out);
    return out;
}

cv::Mat get_projection_matrix() {
    cv::Mat projection_matrix = (cv::Mat_<double>(3, 4) << 
        707.0912, 0, 601.8873, 0, 
        0, 707.0912, 183.1104, 0, 
        0, 0, 1, 0);
    return projection_matrix;
}

bool is_new_keyframe(match_datatype match_data, Keyframe keyframe, int threshold = 100) 
{
    // Check if the new frame is different enough from previous keyframe
    cv::Mat diff;
    cv::absdiff(keyframe.descriptors, match_data.descriptors2, diff);
    int count = cv::countNonZero(diff);
    return count > threshold;
}

cv::Mat get_projection_matrix_for_keyframe(Keyframe keyframe, cv::Mat k_matrix) 
{
    cv::Mat projection_matrix;

    // cv::hconcat(cv::Mat::eye(3,3,CV_64F), -keyframe.translation, projection_matrix);
    // projection_matrix = keyframe.rotation * projection_matrix;
    cv::hconcat(keyframe.rotation, keyframe.translation, projection_matrix);
    projection_matrix = k_matrix * projection_matrix;
    
    return projection_matrix;
}
