

cv::Mat find_fundamental_matrix(match_datatype& match_data) 
{
    std::vector<cv::Point2f> points1, points2;
    for (int i = 0; i < match_data.matches.size(); i++) 
    {
        points1.push_back(match_data.keypoints1[match_data.matches[i][0].queryIdx].pt);
        points2.push_back(match_data.keypoints2[match_data.matches[i][0].trainIdx].pt);
    }
    cv::Mat fundamental_matrix = cv::findFundamentalMat(points1, points2, cv::FM_RANSAC, 3, 0.9);

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

    for (int i = 0; i < points4D.cols; i++)
    {
        point3D_datatype point_data;
        point_data.point3D = cv::Point3f(
            points4D.at<float>(0, i) / points4D.at<float>(3, i),
            points4D.at<float>(1, i) / points4D.at<float>(3, i),
            points4D.at<float>(2, i) / points4D.at<float>(3, i)
        );

        point_data.viewing_direction = cv::Vec3f(point_data.point3D.x, point_data.point3D.y, point_data.point3D.z);

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
        cv::Mat pt2 = projMat2 * pt;
        pt2 /= pt2.at<double>(2);
        cv::Point2f pt2_2d = cv::Point2f(pt2.at<double>(0), pt2.at<double>(1));
        cv::circle(img2, pt2_2d, 3, cv::Scalar(0, 255, 0), -1);
    }
    
    return img1;
}

