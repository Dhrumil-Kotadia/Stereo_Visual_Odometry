#include "im_utils_stereo.hpp"
#include "epipolar_geometry_utils.hpp"
#include "filesystem_utils.hpp"
#include "bundle_adjustment.hpp"

#include <opencv4/opencv2/opencv.hpp>
#include <opencv4/opencv2/highgui.hpp>
#include <opencv4/opencv2/core/eigen.hpp>

#include <iostream>
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

    match_data = match_features(images1[0], images1[1], tools);
    fundamental_matrix = find_fundamental_matrix(match_data);
    detections_img = draw_valid_detections(match_data, images1[1]);
    matches_img = draw_matches(match_data);

    save_image(directories.detections_directory_path + "detections.png", detections_img);
    save_image(directories.matches_directory_path + "matches.png", matches_img);

    for (size_t i = 0; i < match_data.points2f_1.size(); i++) 
    {
        cv::Point2f point1 = match_data.points2f_1[i];
        int disparity = disparity_map.at<short>(point1.y, point1.x);
        float depth = -projection_matrix_2.at<double>(0,3) / disparity;
        float X = (point1.x - projection_matrix_1.at<double>(0,2)) * depth / projection_matrix_1.at<double>(0,0);
        float Y = (point1.y - projection_matrix_1.at<double>(1,2)) * depth / projection_matrix_1.at<double>(1,1);
        point3D_datatype point_data;
        point_data.point3D = cv::Point3f(X, Y, depth);
        match_data.points3D.push_back(point_data);
    }

    cv::Mat triangulation_result;
    std::vector<cv::Point2f> correspondences_2d;
    std::vector<cv::Point3f> correspondences_3d;

    triangulation_result = visualize_triangulation_result(match_data.points3D, images1[0], images1[1], projection_matrix_1, projection_matrix_2);
    save_image(directories.save_directory_path + "triangulation_result.png", triangulation_result);

    keyframe.rotation = r_matrix_1;
    keyframe.translation = t_matrix_1;
    for (int i = 0; i<=match_data.points2f_1.size(); i++)
    {
        if (match_data.points3D[i].point3D.z > 0)
        {
            keyframe.points2f.push_back(match_data.points2f_1[i]);
            keyframe.points3D.push_back(match_data.points3D[i]);
            correspondences_3d.push_back(match_data.points3D[i].point3D);
            correspondences_2d.push_back(match_data.points2f_2[i]);
        }
    }
    keyframes.push_back(keyframe);
    
    std::ofstream file("../outputs/triangulated_points.txt");
    for (const auto& pt : keyframes.back().points3D) 
    {
        file << pt.point3D.x << "," << pt.point3D.z << std::endl;
        if (pt.point3D.z < 0) 
        {
            std::cout << "Before Loop" << pt.point3D << std::endl;
        }
    }
    file.close();

    std::ofstream file2("../outputs/translation.txt");
    file2 << keyframe.translation.at<double>(0,0) << "," << keyframe.translation.at<double>(2,0) << std::endl;
    file2.close();

    cv::Mat rvec, tvec;
    cv::Mat rotation;

    cv::solvePnP(correspondences_3d, correspondences_2d, k_matrix_2, cv::Mat(), rvec, tvec, false, cv::SOLVEPNP_ITERATIVE);
    // cv::solvePnPRefineLM(correspondences_3d, correspondences_2d, k_matrix_2, cv::Mat(), rvec, tvec);

    cv::Rodrigues(rvec, rotation);
    
    keyframe.rotation = rotation;
    keyframe.translation = tvec;
    keyframe.points2f.clear();
    keyframe.points3D.clear();
    for (int i = 0; i< match_data.points2f_2.size(); i++)
    {
        if (match_data.points3D[i].point3D.z > 0)
        {
            keyframe.points2f.push_back(match_data.points2f_2[i]);
            keyframe.points3D.push_back(match_data.points3D[i]);
        }
    }
    keyframes.push_back(keyframe);

    cv::Mat rotation_matrix_global = keyframes.back().rotation;
    cv::Mat translation_matrix_global = keyframes.back().translation;

    for(int i = 1; i < images1.size()-1; i++)
    {
        if(i>8)
        {
            break;
        }
        std::cout << "Processing frame " << i << " and frame " << i+1 << std::endl;
        
        Keyframe new_keyframe;
        std::vector<cv::Point2f> correspondences_2d;
        std::vector<cv::Point3f> correspondences_3d;
        cv::Mat detections_img;
        cv::Mat matches_img;

        match_data = match_features(images1[i], images1[i+1], tools);

        fundamental_matrix = find_fundamental_matrix(match_data);

        get_disparity_map(images1[i+1], images2[i+1], disparity_map, tools);
        detections_img = draw_valid_detections(match_data, images1[i+1]);
        matches_img = draw_matches(match_data);
        
        save_image(directories.detections_directory_path + "detections_" + std::to_string(i+1) + ".png", detections_img);
        save_image(directories.matches_directory_path + "matches_" + std::to_string(i+1) + ".png", matches_img);

        for (int j = 0; j < keyframes.back().points2f.size(); j++)
        {         
            for (int k = 0; k < match_data.points2f_1.size(); k++)
            {
                if (cv::norm(keyframes.back().points2f[j] - match_data.points2f_1[k]) < 1e-3)
                {
                    correspondences_2d.push_back(match_data.points2f_2[k]);
                    correspondences_3d.push_back(keyframes.back().points3D[j].point3D);
                    break;
                }    
            }
        }

        cv::solvePnPRansac(correspondences_3d, correspondences_2d, k_matrix_2, cv::Mat(), rvec, tvec, false, cv::SOLVEPNP_ITERATIVE);

        cv::Rodrigues(rvec, rotation);

        translation_matrix_global = translation_matrix_global + rotation_matrix_global * tvec;
        rotation_matrix_global = rotation_matrix_global * rotation;
        
        new_keyframe.rotation = rotation_matrix_global;
        new_keyframe.translation = translation_matrix_global;

        cv::Mat projection_matrix = get_projection_matrix_for_keyframe(new_keyframe, k_matrix_1);

        for (int j = 0; j < match_data.points2f_2.size(); j++)
        {
            cv::Point2f point1 = match_data.points2f_2[j];
            int disparity = disparity_map.at<short>(point1.y, point1.x);
            float depth = -projection_matrix.at<double>(0,3) / disparity;
            float X = (point1.x - projection_matrix.at<double>(0,2)) * depth / projection_matrix.at<double>(0,0);
            float Y = (point1.y - projection_matrix.at<double>(1,2)) * depth / projection_matrix.at<double>(1,1);
            point3D_datatype point_data;
            point_data.point3D = cv::Point3f(X, Y, depth);
            match_data.points3D.push_back(point_data);
        }

        for (int j = 0; j < match_data.points2f_2.size(); j++)
        {
            if (match_data.points3D[j].point3D.z > 0)
            {
                new_keyframe.points2f.push_back(match_data.points2f_2[j]);
                new_keyframe.points3D.push_back(match_data.points3D[j]);
            }
        }
        
        keyframes.push_back(new_keyframe);

        triangulation_result = visualize_triangulation_result(match_data.points3D, images1[i], images1[i+1], projection_matrix, projection_matrix);
        save_image(directories.save_directory_path + "triangulation_result_" + std::to_string(i+1) + ".png", triangulation_result);

        std::ofstream file("../outputs/triangulated_points.txt", std::ios::app);
        for (const auto& pt : keyframes.back().points3D)
        {
            file << pt.point3D.x << "," << pt.point3D.z << std::endl;
        }
        file.close();

        std::ofstream file2("../outputs/translation.txt", std::ios::app);
        file2 << keyframes.back().translation.at<double>(0) << "," << keyframes.back().translation.at<double>(2) << std::endl;
        file2.close();
    }

    bundle_adjustment(keyframes, k_matrix_1);
    
    for(const auto& keyframe : keyframes) 
    {
        for (const auto& pt : keyframe.points3D) {
            file << pt.point3D.x << "," << pt.point3D.z << std::endl;
        }
    }
    file.close();

    for(const auto& keyframe : keyframes) 
    {
        file2 << keyframe.translation.at<double>(0,0) << "," << keyframe.translation.at<double>(2,0) << std::endl;
    }
    file2.close();


    return 0;
    
}