#include "im_utils_stereo.hpp"
#include "epipolar_geometry_utils.hpp"
#include "filesystem_utils.hpp"
#include "bundle_adjustment.hpp"

#include <opencv4/opencv2/opencv.hpp>
#include <opencv4/opencv2/highgui.hpp>
#include <opencv4/opencv2/core/eigen.hpp>

#include <iostream>
#include <string>
#include <fstream>
#include <Eigen/Dense>

int main() 
{
    std::cout << "Initializing..." << std::endl;
    
    directory_datatype directories = create_directories();
    std::vector<std::string> image_paths;
    std::vector<cv::Mat> images1, images2;
    std::vector<Keyframe> keyframes;
    Keyframe keyframe;
    im_tools tools;
    cv::Mat k_matrix_1, r_matrix_1, t_matrix_1;
    cv::Mat k_matrix_2, r_matrix_2, t_matrix_2;
    bool use_pnp = false;
    bool use_fundamental_matrix = true;

    image_paths = read_directory(directories.read_directory_path_1);
    std::sort(image_paths.begin(), image_paths.end());

    for (auto path : image_paths) 
    {
        cv::Mat img = read_image(path);
        images1.push_back(img);
    }

    image_paths = read_directory(directories.read_directory_path_2);
    std::sort(image_paths.begin(), image_paths.end());
    for (auto path : image_paths) 
    {
        cv::Mat img = read_image(path);
        images2.push_back(img);
    }

    cv::Mat projection_matrix_1 = (cv::Mat_<double>(3, 4) << 
    707.0912, 0, 601.8873, 0, 
    0, 707.0912, 183.1104, 0, 
    0, 0, 1, 0);
    cv::Mat projection_matrix_2 = (cv::Mat_<double>(3, 4) <<
    707.0912, 0, 601.8873, -379.8145,
    0, 707.0912, 183.1104, 0,
    0, 0, 1, 0);

    cv::decomposeProjectionMatrix(projection_matrix_1, k_matrix_1, r_matrix_1, t_matrix_1);
    cv::decomposeProjectionMatrix(projection_matrix_2, k_matrix_2, r_matrix_2, t_matrix_2);

    t_matrix_1 = t_matrix_1.rowRange(0, 3) / t_matrix_1.at<double>(3);
    t_matrix_2 = t_matrix_2.rowRange(0, 3) / t_matrix_2.at<double>(3);

    cv::Mat image_l = images1[0];
    cv::Mat image_r = images2[0];
    cv::Mat disparity_map;
    match_datatype match_data;
    cv::Mat fundamental_matrix;
    cv::Mat detections_img;
    cv::Mat matches_img;

    get_disparity_map(image_l, image_r, disparity_map, tools);
    disparity_map = disparity_map / 16.0;

    match_data = match_features(image_l, image_r, tools);
    fundamental_matrix = find_fundamental_matrix(match_data);
    detections_img = draw_valid_detections(match_data, image_r);
    matches_img = draw_matches(match_data);

    save_image(directories.detections_directory_path + "detections_0.png", detections_img);
    save_image(directories.matches_directory_path + "matches_0.png", matches_img);

    for (size_t i = 0; i < match_data.points2f_1.size(); i++) 
    {
        cv::Point2f point1 = match_data.points2f_1[i];
        float disparity = disparity_map.at<short>(point1.y, point1.x);
        float depth = 379.815 / disparity;
        float X = (point1.x - projection_matrix_1.at<double>(0,2)) * depth / projection_matrix_1.at<double>(0,0);
        float Y = (point1.y - projection_matrix_1.at<double>(1,2)) * depth / projection_matrix_1.at<double>(1,1);
        point3D_datatype point_data;
        point_data.point3D = cv::Point3f(X, Y, depth);
        match_data.points3D.push_back(point_data);
    }

    cv::Mat triangulation_result;
    std::vector<cv::Point2f> correspondences_2d;
    std::vector<cv::Point3f> correspondences_3d;

    triangulation_result = visualize_triangulation_result(match_data.points3D, image_l, image_r, projection_matrix_1, projection_matrix_2);
    save_image(directories.save_directory_path + "triangulation_result_0.png", triangulation_result);

    keyframe.rotation = r_matrix_1;
    keyframe.translation = t_matrix_1;

    for (int i = 0; i< match_data.points2f_1.size(); i++)
    {
        if (match_data.points3D[i].point3D.z > 0)
        {
            keyframe.points2f.push_back(match_data.points2f_1[i]);
            keyframe.points3D.push_back(match_data.points3D[i]);
        }
    }
    std::cout << "Number of 3D points: " << keyframe.points3D.size() << std::endl;
    std::cout << "Number of 2D points: " << keyframe.points2f.size() << std::endl;
    keyframes.push_back(keyframe);

    cv::Mat rvec, tvec;
    cv::Mat rotation;

    cv::Mat rotation_matrix_global = r_matrix_1;
    cv::Mat translation_matrix_global = t_matrix_1;

    // make transformation matrix from rotation and translation
    cv::Mat transformation_matrix = cv::Mat::eye(4, 4, CV_64F);
    rotation_matrix_global.copyTo(transformation_matrix.rowRange(0, 3).colRange(0, 3));
    translation_matrix_global.copyTo(transformation_matrix.rowRange(0, 3).col(3));


    std::ofstream points_file("../outputs/triangulated_points.txt");
    std::ofstream translation_file("../outputs/translation.txt");

    for(int i = 0; i < images1.size()-1; i++)
    {
        if(i>25)
        {
            break;
        }
        std::cout << "Processing frame " << i << " and frame " << i+1 << std::endl;
        
        Keyframe new_keyframe;
        std::vector<cv::Point2f> correspondences_2d;
        std::vector<cv::Point3f> correspondences_3d;
        cv::Mat detections_img;
        cv::Mat matches_img;

        image_l = images1[i+1];
        image_r = images2[i+1];

        match_data = match_features(images1[i], images1[i+1], tools);
        std::cout << "Features matched" << std::endl;

        fundamental_matrix = find_fundamental_matrix(match_data);
        std::cout << "Fundamental matrix found" << std::endl;
        detections_img = draw_valid_detections(match_data, image_l);
        save_image(directories.detections_directory_path + "detections_" + std::to_string(i+1) + ".png", detections_img);
        matches_img = draw_matches(match_data);
        save_image(directories.matches_directory_path + "matches_" + std::to_string(i+1) + ".png", matches_img);        

        if (use_pnp)
        {
            for (int j = 0; j < keyframes.back().points2f.size(); j++)
            {         
                for (int k = 0; k < match_data.points2f_1.size(); k++)
                {
                    if (cv::norm(keyframes.back().points2f[j] - match_data.points2f_1[k]) < 1)
                    {
                        correspondences_2d.push_back(match_data.points2f_2[k]);
                        correspondences_3d.push_back(keyframes.back().points3D[j].point3D);
                        break;
                    }    
                }
            }

            if (correspondences_2d.size() < 10)
            {
                std::cout << "Not enough correspondences for PnP" << std::endl;
                std::cout << "Number of correspondences: " << correspondences_2d.size() << std::endl;
                continue;
            }
            cv::solvePnPRansac(correspondences_3d, correspondences_2d, k_matrix_1, cv::Mat(), rvec, tvec, false, cv::SOLVEPNP_ITERATIVE);
            std::cout << "TVEC: " << tvec << std::endl;
            cv::Rodrigues(rvec, rotation);

            cv::Mat local_transformation_matrix = cv::Mat::eye(4, 4, CV_64F);
            rotation.copyTo(local_transformation_matrix.rowRange(0, 3).colRange(0, 3));
            tvec.copyTo(local_transformation_matrix.rowRange(0, 3).col(3));

            transformation_matrix = transformation_matrix * local_transformation_matrix;
        }
        else if (use_fundamental_matrix)
        {
            cv::Mat essential_matrix = find_essential_matrix(fundamental_matrix, k_matrix_1);
            std::cout << "Essential matrix found" << std::endl;
            cv::Mat rotation_matrix, translation_matrix;
            cv::recoverPose(essential_matrix, match_data.points2f_1, match_data.points2f_2, k_matrix_1, rotation_matrix, translation_matrix);
            std::cout << "Pose recovered" << std::endl;
            // Use pnp to refine the transformation

            cv::Mat rvec, tvec;
            cv::Rodrigues(rotation_matrix, rvec);
            std::cout << "Checking 2D-3D correspondences" << std::endl;
            
            tvec = translation_matrix;
            for (int j = 0; j < keyframes.back().points2f.size(); j++)
            {         
                for (int k = 0; k < match_data.points2f_1.size(); k++)
                {
                    if (cv::norm(keyframes.back().points2f[j] - match_data.points2f_1[k]) < 1)
                    {
                        correspondences_2d.push_back(match_data.points2f_2[k]);
                        correspondences_3d.push_back(keyframes.back().points3D[j].point3D);
                        break;
                    }    
                }
            }

            if (correspondences_2d.size() < 10)
            {
                std::cout << "Not enough correspondences for PnP Refinement" << std::endl;
            }
            else
            {
                cv::solvePnPRefineLM(correspondences_3d, correspondences_2d, k_matrix_1, cv::Mat(), rvec, tvec);
                cv::Rodrigues(rvec, rotation_matrix);
                translation_matrix = tvec;
                // std::cout << "translation after pnp: " << translation_matrix << std::endl;
            }
            cv::Mat local_transformation_matrix = cv::Mat::eye(4, 4, CV_64F);
            rotation_matrix.copyTo(local_transformation_matrix.rowRange(0, 3).colRange(0, 3));
            translation_matrix.copyTo(local_transformation_matrix.rowRange(0, 3).col(3));

            transformation_matrix = transformation_matrix * local_transformation_matrix;
        }

        new_keyframe.rotation = transformation_matrix.rowRange(0, 3).colRange(0, 3);
        new_keyframe.translation = transformation_matrix.rowRange(0, 3).col(3);

        std::cout << "Translation of frame " << i+1 << ": " << new_keyframe.translation << std::endl;

        get_projection_matrix_for_keyframe(new_keyframe, k_matrix_1, projection_matrix_1, projection_matrix_2);        

        get_disparity_map(image_l, image_r, disparity_map, tools);
        disparity_map = disparity_map / 16.0;

        cv::Mat transformation_matrix_inv;
        cv::invert(transformation_matrix, transformation_matrix_inv);

        for (int j = 0; j < match_data.points2f_2.size(); j++)
        {
            cv::Point2f point1 = match_data.points2f_2[j];
            float disparity = disparity_map.at<short>(point1.y, point1.x);
            float depth = 379.815 / disparity;
            float X = (point1.x - k_matrix_1.at<double>(0,2)) * depth / k_matrix_1.at<double>(0,0);
            float Y = (point1.y - k_matrix_1.at<double>(1,2)) * depth / k_matrix_1.at<double>(1,1);
            cv::Mat point3D = (cv::Mat_<double>(4, 1) << X, Y, depth, 1);
            cv::Mat point3D_global = transformation_matrix_inv * point3D;
            X = point3D_global.at<double>(0);
            Y = point3D_global.at<double>(1);
            depth = point3D_global.at<double>(2);
            point3D_datatype point_data;
            point_data.point3D = cv::Point3f(X, Y, depth);
            match_data.points3D.push_back(point_data);
        }
        

        for (int j = 0; j < match_data.points2f_2.size(); j++)
        {
            new_keyframe.points2f.push_back(match_data.points2f_2[j]);
            new_keyframe.points3D.push_back(match_data.points3D[j]);
        }

        keyframes.push_back(new_keyframe);

        triangulation_result = visualize_triangulation_result(keyframes.back().points3D, images1[i], images1[i], projection_matrix_1, projection_matrix_1);
        save_image(directories.save_directory_path + "triangulation_result_" + std::to_string(i+1) + ".png", triangulation_result);
        
        points_file.open("../outputs/triangulated_points.txt", std::ios_base::app);
        for (const auto& pt : keyframes.back().points3D)
        {
            points_file << pt.point3D.x << "," << pt.point3D.y << "," << pt.point3D.z << std::endl;
        }
        points_file.close();

        translation_file.open("../outputs/translation.txt", std::ios_base::app);
        translation_file << keyframes.back().translation.at<double>(0) << "," << keyframes.back().translation.at<double>(1) << "," << keyframes.back().translation.at<double>(2) << std::endl;
        translation_file.close();
    }

    bundle_adjustment(keyframes, k_matrix_1);
    
    for(const auto& keyframe : keyframes) 
    {
        for (const auto& pt : keyframe.points3D) {
            points_file << pt.point3D.x << "," << pt.point3D.z << std::endl;
        }
    }
    points_file.close();

    for(const auto& keyframe : keyframes) 
    {
        translation_file << keyframe.translation.at<double>(0,0) << "," << keyframe.translation.at<double>(2,0) << std::endl;
    }
    translation_file.close();


    return 0;
    
}