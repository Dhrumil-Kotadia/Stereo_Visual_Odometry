#include <opencv4/opencv2/opencv.hpp>
#include <opencv4/opencv2/highgui.hpp>
#include <vector>
#include <Eigen/Dense>

struct im_tools
{
    cv::Ptr<cv::Feature2D> detector = cv::ORB::create(100);
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
    std::vector<point3D_datatype> map_points; 
};

void update_keyframe(Keyframe &keyframe, cv::Mat rotation, cv::Mat translation, std::vector<cv::KeyPoint> keypoints, std::vector<cv::Point2f> points2f, cv::Mat descriptors, std::vector<point3D_datatype> map_points) 
{
    keyframe.rotation = rotation.clone();
    keyframe.translation = translation.clone();
    keyframe.keypoints = keypoints;
    keyframe.points2f = points2f;
    keyframe.descriptors = descriptors;
    keyframe.map_points = map_points;
}

cv::Mat read_image(const std::string &path) 
{
    cv::Mat img = cv::imread(path);
    if (img.empty()) {
        std::cerr << "Error: Image not found at " << path << std::endl;
        exit(1);
    }
    return img;
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
    std::cout << "Points2f_1 size: " << match_data.points2f_1.size() << std::endl;
    //draw circles using points2f_1
    for (int i = 0; i < match_data.points2f_1.size(); i++)
    {
        cv::circle(out, match_data.points2f_1[i], 3, cv::Scalar(255, 0, 0), -1);
    }
    return out;

}

void save_image(const std::string path, const cv::Mat img) 
{
    cv::imwrite(path, img);
}

match_datatype match_features(const cv::Mat img1, const cv::Mat img2, im_tools tools) 
{
    feature_datatype feature_data_1 = extract_features(img1, tools.detector);
    feature_datatype feature_data_2 = extract_features(img2, tools.detector);
    // std::cout << "Features Extracted" << std::endl;
    std::vector<std::vector<cv::DMatch>> matches;
    tools.matcher->knnMatch(feature_data_1.descriptors, feature_data_2.descriptors, matches, 3000);
    // std::cout << "Matches Found" << std::endl;
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

cv::Mat find_fundamental_matrix(match_datatype& match_data) 
{
    std::vector<cv::Point2f> points1, points2;
    for (int i = 0; i < match_data.matches.size(); i++) 
    {
        points1.push_back(match_data.keypoints1[match_data.matches[i][0].queryIdx].pt);
        points2.push_back(match_data.keypoints2[match_data.matches[i][0].trainIdx].pt);
    }
    cv::Mat fundamental_matrix = cv::findFundamentalMat(points1, points2, cv::FM_RANSAC, 3, 0.999);

    cv::SVD svd(fundamental_matrix);
    cv::Mat W = cv::Mat::zeros(3, 3, CV_64F);
    W.at<double>(0, 0) = svd.w.at<double>(0, 0);
    W.at<double>(1, 1) = svd.w.at<double>(1, 0);

    // std::cout << "Fundamental Matrix: " << std::endl << fundamental_matrix << std::endl;

    cv::Mat fundamental_matrix_clean = svd.u * W * svd.vt;
    // std::cout << "Fundamental Matrix Clean: " << std::endl << fundamental_matrix_clean << std::endl;

    // remove outliers based on fundamental matrix and update keypoints and points2f
    std::vector<cv::Point2f> points1_inliers, points2_inliers;
    std::vector<std::vector<cv::DMatch>> matches_inliers;
    for (int i = 0; i < points1.size(); i++) 
    {
        cv::Mat point1 = (cv::Mat_<double>(3, 1) << points1[i].x, points1[i].y, 1);
        cv::Mat point2 = (cv::Mat_<double>(3, 1) << points2[i].x, points2[i].y, 1);
        cv::Mat epipolar_line = fundamental_matrix_clean * point1;
        double error = point2.dot(epipolar_line);
        if (std::abs(error) < 0.05) 
        {
            points1_inliers.push_back(points1[i]);
            points2_inliers.push_back(points2[i]);
            // Update matches_inliers
            matches_inliers.push_back(match_data.matches[i]);
        }
    }

    // cv::Mat fundamental_matrix_clean_inliers = cv::findFundamentalMat(points1_inliers, points2_inliers, cv::FM_8POINT);

    // update match_data with inliers
    match_data.points2f_1 = points1_inliers;
    match_data.points2f_2 = points2_inliers;
    match_data.matches = matches_inliers;

    return fundamental_matrix_clean;
}

cv::Mat find_essential_matrix(const cv::Mat &fundamental_matrix, const cv::Mat &k_matrix) 
{
    cv::Mat essential_matrix = k_matrix.t() * fundamental_matrix * k_matrix;
    return essential_matrix;
}

cv::Mat get_projection_matrix() {
    cv::Mat projection_matrix = (cv::Mat_<double>(3, 4) << 
        707.0912, 0, 601.8873, 0, 
        0, 707.0912, 183.1104, 0, 
        0, 0, 1, 0);
    return projection_matrix;
}

void display_epipolar_constraints_error(const cv::Mat &fundamental_matrix, const match_datatype &match_data) 
{
    // check epipolar constraints
    double error = 0;
    for (int i = 0; i < match_data.points2f_1.size(); i++) 
    {
        cv::Mat point1 = (cv::Mat_<double>(3, 1) << match_data.points2f_1[i].x, match_data.points2f_1[i].y, 1);
        cv::Mat point2 = (cv::Mat_<double>(3, 1) << match_data.points2f_2[i].x, match_data.points2f_2[i].y, 1);
        cv::Mat epipolar_line = fundamental_matrix * point1;
        error += point2.dot(epipolar_line);
    }    
    std::cout <<"Average Epipolar Constraint Error: " << error / match_data.points2f_1.size() << std::endl; 
}

void triangulate_and_store(match_datatype &match_data, const cv::Mat &projMat1, const cv::Mat &projMat2)
{
    cv::Mat points4D;
    cv::triangulatePoints(projMat1, projMat2, match_data.points2f_1, match_data.points2f_2, points4D);

    // non linear triangulation


    for (int i = 0; i < points4D.cols; i++)
    {
        // remove points behind the camera
        // if (points4D.at<float>(2, i) > 0) 
        // {
        //     continue;
        // }
        point3D_datatype point_data;
        point_data.point3D = cv::Point3f(
            points4D.at<float>(0, i) / points4D.at<float>(3, i),
            points4D.at<float>(1, i) / points4D.at<float>(3, i),
            points4D.at<float>(2, i) / points4D.at<float>(3, i)
        );

        // Compute viewing direction (camera center to 3D point)
        point_data.viewing_direction = cv::Vec3f(point_data.point3D.x, point_data.point3D.y, point_data.point3D.z);

        // Find best descriptor with minimum Hamming distance
        int best_idx = -1;
        int min_distance = INT_MAX;
        for (size_t j = 0; j < match_data.matches[i].size(); j++)
        {
            int query_idx = match_data.matches[i][j].queryIdx;
            int train_idx = match_data.matches[i][j].trainIdx;
            int distance = match_data.matches[i][j].distance;

            if (distance < min_distance)
            {
                min_distance = distance;
                best_idx = train_idx;
            }
        }

        point_data.best_descriptor_idx = best_idx;

        // Compute visibility range (dmin, dmax)
        float depth = point_data.point3D.z;
        point_data.dmin = depth * 0.8; // Example: 80% of depth as min
        point_data.dmax = depth * 1.2; // Example: 120% of depth as max

        // Store the structured point in match_data
        match_data.points3D.push_back(point_data);
    }
}

cv::Mat visualize_triangulation_result(std::vector<point3D_datatype> point_data, cv::Mat img1, cv::Mat img2, cv::Mat projMat1, cv::Mat projMat2) 
{
    for (size_t i = 0; i < point_data.size(); i++) 
    {
        cv::Mat pt = (cv::Mat_<double>(4, 1) << point_data[i].point3D.x, point_data[i].point3D.y, point_data[i].point3D.z, 1);
        cv::Mat pt1 = projMat1 * pt;
        // cv::Mat pt2 = projMat2 * pt;
        pt1 /= pt1.at<double>(2);
        // pt2 /= pt2.at<double>(2);
        cv::Point2f pt1_2d = cv::Point2f(pt1.at<double>(0), pt1.at<double>(1));
        // cv::Point2f pt2_2d = cv::Point2f(pt2.at<double>(0), pt2.at<double>(1));
        cv::circle(img1, pt1_2d, 3, cv::Scalar(0, 255, 0), -1);
        // cv::circle(img2, pt2_2d, 3, cv::Scalar(0, 255, 0), -1);

        // // print image coordinates on terminal if it is within image bounds
        // if (pt1_2d.x >= 0 && pt1_2d.x < img1.cols && pt1_2d.y >= 0 && pt1_2d.y < img1.rows) 
        // {
        //     std::cout << "Image 1: " << pt1_2d << std::endl;
        //     std::cout << "3D Point: " << point_data[i].point3D << std::endl;
        // }

    }
    
    return img1;
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

void extract_pose(const cv::Mat& E, std::vector<cv::Mat>& R, std::vector<cv::Mat>& T) 
    {
        cv::SVD svd(E, cv::SVD::FULL_UV);
        cv::Mat U = svd.u;
        cv::Mat V = svd.vt.t();
        
        cv::Mat W = (cv::Mat_<double>(3,3) << 0, -1, 0,
                                             1,  0, 0,
                                             0,  0, 1);
        
        cv::Mat R1 = U * W * V.t();
        cv::Mat R2 = R1;
        cv::Mat R3 = U * W.t() * V.t();
        cv::Mat R4 = R3;
        
        cv::Mat T1 = U.col(2);
        cv::Mat T2 = -U.col(2);
        cv::Mat T3 = U.col(2);
        cv::Mat T4 = -U.col(2);
        
        R = {R1, R2, R3, R4};
        T = {T1, T2, T3, T4};
        
        for (size_t i = 0; i < R.size(); ++i) {
            if (cv::determinant(R[i]) < 0) {
                R[i] = -R[i];
                T[i] = -T[i];
            }
        }
    }