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
#include <algorithm>
#include <Eigen/Dense>

std::vector<cv::Mat> loadImages(const std::string &directoryPath);
Keyframe createInitialKeyframe(const cv::Mat &leftImage, const cv::Mat &rightImage,
                               const cv::Mat &projMatrix1, const cv::Mat &projMatrix2,
                               const directory_datatype &directories, im_tools &tools,
                               cv::Mat &kMatrix, cv::Mat &rMatrix, cv::Mat &tMatrix);
Keyframe processFrame(int frameIndex,
                      const std::vector<cv::Mat> &images1,
                      const std::vector<cv::Mat> &images2,
                      const cv::Mat &kMatrix,
                      const cv::Mat &projMatrix1, const cv::Mat &projMatrix2,
                      const Keyframe &prevKeyframe,
                      im_tools &tools,
                      const directory_datatype &directories,
                      bool usePnP, bool useFundamental,
                      cv::Mat &transformationMatrix);
void savePointsAndTranslations(const std::vector<Keyframe> &keyframes,
                               const std::string &pointsFilePath,
                               const std::string &translationFilePath);

int main() {
    std::cout << "Initializing..." << std::endl;

    directory_datatype directories = create_directories();
    std::vector<cv::Mat> images1 = loadImages(directories.read_directory_path_1);
    std::vector<cv::Mat> images2 = loadImages(directories.read_directory_path_2);

    cv::Mat projectionMatrix1 = (cv::Mat_<double>(3, 4) <<
                                 707.0912, 0, 601.8873, 0,
                                 0, 707.0912, 183.1104, 0,
                                 0, 0, 1, 0);
    cv::Mat projectionMatrix2 = (cv::Mat_<double>(3, 4) <<
                                 707.0912, 0, 601.8873, -379.8145,
                                 0, 707.0912, 183.1104, 0,
                                 0, 0, 1, 0);

    cv::Mat kMatrix1, rMatrix1, tMatrix1;
    cv::Mat kMatrix2, rMatrix2, tMatrix2;
    cv::decomposeProjectionMatrix(projectionMatrix1, kMatrix1, rMatrix1, tMatrix1);
    cv::decomposeProjectionMatrix(projectionMatrix2, kMatrix2, rMatrix2, tMatrix2);
    tMatrix1 = tMatrix1.rowRange(0, 3) / tMatrix1.at<double>(3);
    tMatrix2 = tMatrix2.rowRange(0, 3) / tMatrix2.at<double>(3);

    im_tools tools;
    bool usePnP = false;
    bool useFundamental = true;

    Keyframe initialKeyframe = createInitialKeyframe(images1[0], images2[0],
                                                     projectionMatrix1, projectionMatrix2,
                                                     directories, tools,
                                                     kMatrix1, rMatrix1, tMatrix1);
    std::vector<Keyframe> keyframes;
    keyframes.push_back(initialKeyframe);

    cv::Mat transformationMatrix = cv::Mat::eye(4, 4, CV_64F);
    rMatrix1.copyTo(transformationMatrix.rowRange(0, 3).colRange(0, 3));
    tMatrix1.copyTo(transformationMatrix.rowRange(0, 3).col(3));

    std::ofstream pointsFile("../outputs/triangulated_points.txt", std::ios::app);
    std::ofstream translationFile("../outputs/translation.txt", std::ios::app);
    if (!pointsFile.is_open() || !translationFile.is_open()) {
        std::cerr << "Error: Unable to open output files." << std::endl;
        return -1;
    }

    int numFrames = std::min(static_cast<int>(images1.size()), 26);
    for (int i = 0; i < numFrames - 1; i++) {
        std::cout << "Processing frame " << i << " and frame " << i + 1 << std::endl;
        Keyframe newKeyframe = processFrame(i, images1, images2, kMatrix1,
                                            projectionMatrix1, projectionMatrix2,
                                            keyframes.back(), tools, directories,
                                            usePnP, useFundamental, transformationMatrix);
        keyframes.push_back(newKeyframe);

        cv::Mat triangulationResult = visualize_triangulation_result(newKeyframe.points3D,
                                                                     images1[i],
                                                                     images1[i],
                                                                     projectionMatrix1,
                                                                     projectionMatrix1);
        std::string triangulationFile = directories.save_directory_path +
                                          "triangulation_result_" + std::to_string(i+1) + ".png";
        save_image(triangulationFile, triangulationResult);

        for (const auto &pt : newKeyframe.points3D) {
            pointsFile << pt.point3D.x << "," << pt.point3D.y << "," << pt.point3D.z << std::endl;
        }
        translationFile << newKeyframe.translation.at<double>(0) << ","
                        << newKeyframe.translation.at<double>(1) << ","
                        << newKeyframe.translation.at<double>(2) << std::endl;
    }
    pointsFile.close();
    translationFile.close();

    bundle_adjustment(keyframes, kMatrix1);

    savePointsAndTranslations(keyframes, "../outputs/triangulated_points.txt", "../outputs/translation.txt");

    return 0;
}


std::vector<cv::Mat> loadImages(const std::string &directoryPath) 
{
    std::vector<std::string> imagePaths = read_directory(directoryPath);
    std::sort(imagePaths.begin(), imagePaths.end());
    std::vector<cv::Mat> images;
    for (const auto &path : imagePaths) 
    {
        cv::Mat img = read_image(path);
        images.push_back(img);
    }
    return images;
}

Keyframe createInitialKeyframe(const cv::Mat &leftImage, const cv::Mat &rightImage,
                               const cv::Mat &projMatrix1, const cv::Mat &projMatrix2,
                               const directory_datatype &directories, im_tools &tools,
                               cv::Mat &kMatrix, cv::Mat &rMatrix, cv::Mat &tMatrix) 
{
    Keyframe keyframe;
    cv::Mat disparityMap;
    match_datatype matchData;

    get_disparity_map(leftImage, rightImage, disparityMap, tools);
    disparityMap = disparityMap / 16.0;

    matchData = match_features(leftImage, rightImage, tools);
    cv::Mat fundamentalMatrix = find_fundamental_matrix(matchData);

    cv::Mat detectionsImg = draw_valid_detections(matchData, rightImage);
    cv::Mat matchesImg = draw_matches(matchData);
    save_image(directories.detections_directory_path + "detections_0.png", detectionsImg);
    save_image(directories.matches_directory_path + "matches_0.png", matchesImg);

    for (size_t i = 0; i < matchData.points2f_1.size(); i++) 
    {
        cv::Point2f pt = matchData.points2f_1[i];
        float disparity = disparityMap.at<short>(pt.y, pt.x);
        float depth = 379.815f / disparity;
        float X = (pt.x - projMatrix1.at<double>(0, 2)) * depth / projMatrix1.at<double>(0, 0);
        float Y = (pt.y - projMatrix1.at<double>(1, 2)) * depth / projMatrix1.at<double>(1, 1);
        point3D_datatype pointData;
        pointData.point3D = cv::Point3f(X, Y, depth);
        matchData.points3D.push_back(pointData);
    }

    // Set the keyframe pose.
    keyframe.rotation = rMatrix;
    keyframe.translation = tMatrix;

    // Store valid correspondences.
    for (size_t i = 0; i < matchData.points2f_1.size(); i++) 
    {
        if (matchData.points3D[i].point3D.z > 0)
        {
            keyframe.points2f.push_back(matchData.points2f_1[i]);
            keyframe.points3D.push_back(matchData.points3D[i]);
        }
    }

    std::cout << "Initial keyframe: " << keyframe.points3D.size()
              << " valid 3D points, " << keyframe.points2f.size() << " 2D points." << std::endl;
    return keyframe;
}

// -----------------------------------------------------------------------------
// Processes a subsequent frame to produce a new keyframe.
Keyframe processFrame(int frameIndex,
                      const std::vector<cv::Mat> &images1,
                      const std::vector<cv::Mat> &images2,
                      const cv::Mat &kMatrix,
                      const cv::Mat &projMatrix1, const cv::Mat &projMatrix2,
                      const Keyframe &prevKeyframe,
                      im_tools &tools,
                      const directory_datatype &directories,
                      bool usePnP, bool useFundamental,
                      cv::Mat &transformationMatrix) 
{
    Keyframe newKeyframe;
    std::vector<cv::Point2f> correspondences2D;
    std::vector<cv::Point3f> correspondences3D;
    cv::Mat detectionsImg, matchesImg, disparityMap;
    match_datatype matchData;
    cv::Mat rvec, tvec, rotation;

    // Match features between previous and current left images.
    cv::Mat prevImage = images1[frameIndex];
    cv::Mat currImage = images1[frameIndex + 1];
    cv::Mat currRightImage = images2[frameIndex + 1];
    matchData = match_features(prevImage, currImage, tools);
    std::cout << "Features matched for frame " << frameIndex + 1 << std::endl;

    // Compute the fundamental matrix.
    cv::Mat fundamentalMatrix = find_fundamental_matrix(matchData);
    std::cout << "Fundamental matrix computed for frame " << frameIndex + 1 << std::endl;

    // Save detections and matches visualization.
    detectionsImg = draw_valid_detections(matchData, currImage);
    save_image(directories.detections_directory_path + "detections_" + std::to_string(frameIndex + 1) + ".png", detectionsImg);
    matchesImg = draw_matches(matchData);
    save_image(directories.matches_directory_path + "matches_" + std::to_string(frameIndex + 1) + ".png", matchesImg);

    // Use either PnP or epipolar geometry to update the camera pose.
    if (usePnP) 
    {
        for (size_t j = 0; j < prevKeyframe.points2f.size(); j++) 
        {
            for (size_t k = 0; k < matchData.points2f_1.size(); k++) 
            {
                if (cv::norm(prevKeyframe.points2f[j] - matchData.points2f_1[k]) < 1) 
                {
                    correspondences2D.push_back(matchData.points2f_2[k]);
                    correspondences3D.push_back(prevKeyframe.points3D[j].point3D);
                    break;
                }
            }
        }

        if (correspondences2D.size() < 4) 
        {
            std::cerr << "Not enough correspondences for PnP in frame " << frameIndex + 1 << std::endl;
        } 
        else 
        {
            cv::solvePnPRansac(correspondences3D, correspondences2D, kMatrix, cv::Mat(),
                               rvec, tvec, false, cv::SOLVEPNP_ITERATIVE);
            std::cout << "PnP solved for frame " << frameIndex + 1 << std::endl;
            cv::Rodrigues(rvec, rotation);

            cv::Mat localTransformation = cv::Mat::eye(4, 4, CV_64F);
            rotation.copyTo(localTransformation.rowRange(0, 3).colRange(0, 3));
            tvec.copyTo(localTransformation.rowRange(0, 3).col(3));
            transformationMatrix = transformationMatrix * localTransformation;
        }
    }
    else if (useFundamental) 
    {
        cv::Mat essentialMatrix = find_essential_matrix(fundamentalMatrix, kMatrix);
        std::cout << "Essential matrix computed for frame " << frameIndex + 1 << std::endl;
        cv::Mat rotationMatrix, translationMatrix;
        cv::recoverPose(essentialMatrix, matchData.points2f_1, matchData.points2f_2,
                        kMatrix, rotationMatrix, translationMatrix);
        std::cout << "Pose recovered for frame " << frameIndex + 1 << std::endl;

        cv::Rodrigues(rotationMatrix, rvec);
        tvec = translationMatrix;
        for (size_t j = 0; j < prevKeyframe.points2f.size(); j++) 
        {
            for (size_t k = 0; k < matchData.points2f_1.size(); k++) 
            {
                if (cv::norm(prevKeyframe.points2f[j] - matchData.points2f_1[k]) < 1) 
                {
                    correspondences2D.push_back(matchData.points2f_2[k]);
                    correspondences3D.push_back(prevKeyframe.points3D[j].point3D);
                    break;
                }
            }
        }

        if (correspondences2D.size() >= 1000) 
        {
            cv::solvePnPRefineLM(correspondences3D, correspondences2D, kMatrix, cv::Mat(), rvec, tvec);
            cv::Rodrigues(rvec, rotationMatrix);
            translationMatrix = tvec;
        } 
        else 
        {
            std::cerr << "Not enough correspondences for PnP refinement in frame " << frameIndex + 1 << std::endl;
        }

        cv::Mat localTransformation = cv::Mat::eye(4, 4, CV_64F);
        rotationMatrix.copyTo(localTransformation.rowRange(0, 3).colRange(0, 3));
        translationMatrix.copyTo(localTransformation.rowRange(0, 3).col(3));
        transformationMatrix = transformationMatrix * localTransformation;
    }

    newKeyframe.rotation = transformationMatrix.rowRange(0, 3).colRange(0, 3);
    newKeyframe.translation = transformationMatrix.rowRange(0, 3).col(3);
    std::cout << "Frame " << frameIndex + 1 << " translation: " << newKeyframe.translation << std::endl;

    get_projection_matrix_for_keyframe(newKeyframe, kMatrix, projMatrix1, projMatrix2);

    get_disparity_map(currImage, currRightImage, disparityMap, tools);
    disparityMap = disparityMap / 16.0;

    // Transform 2D points to 3D in the global coordinate system.
    cv::Mat transformationMatrixInv;
    cv::invert(transformationMatrix, transformationMatrixInv);
    for (size_t j = 0; j < matchData.points2f_2.size(); j++) 
    {
        cv::Point2f pt = matchData.points2f_2[j];
        float disparity = disparityMap.at<short>(pt.y, pt.x);
        float depth = 379.815f / disparity;
        float X = (pt.x - kMatrix.at<double>(0, 2)) * depth / kMatrix.at<double>(0, 0);
        float Y = (pt.y - kMatrix.at<double>(1, 2)) * depth / kMatrix.at<double>(1, 1);
        cv::Mat point3DMat = (cv::Mat_<double>(4, 1) << X, Y, depth, 1);
        cv::Mat point3DGlobal = transformationMatrixInv * point3DMat;
        X = point3DGlobal.at<double>(0);
        Y = point3DGlobal.at<double>(1);
        depth = point3DGlobal.at<double>(2);
        point3D_datatype pointData;
        pointData.point3D = cv::Point3f(X, Y, depth);
        matchData.points3D.push_back(pointData);
    }

    // Assign updated 2D and 3D points to the new keyframe.
    for (size_t j = 0; j < matchData.points2f_2.size(); j++) 
    {
        newKeyframe.points2f.push_back(matchData.points2f_2[j]);
        newKeyframe.points3D.push_back(matchData.points3D[j]);
    }
    return newKeyframe;
}

void savePointsAndTranslations(const std::vector<Keyframe> &keyframes,
                               const std::string &pointsFilePath,
                               const std::string &translationFilePath) 
{
    std::ofstream pointsFile(pointsFilePath, std::ios::app);
    std::ofstream translationFile(translationFilePath, std::ios::app);
    if (!pointsFile.is_open() || !translationFile.is_open()) 
    {
        std::cerr << "Error: Unable to open output files for final save." << std::endl;
        return;
    }
    for (const auto &keyframe : keyframes) 
    {
        for (const auto &pt : keyframe.points3D) 
        {
            pointsFile << pt.point3D.x << "," << pt.point3D.y << "," << pt.point3D.z << std::endl;
        }
        translationFile << keyframe.translation.at<double>(0) << ","
                        << keyframe.translation.at<double>(1) << ","
                        << keyframe.translation.at<double>(2) << std::endl;
    }
    pointsFile.close();
    translationFile.close();
}
