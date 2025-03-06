#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <opencv4/opencv2/opencv.hpp>
#include <opencv4/opencv2/highgui.hpp>
#include <opencv4/opencv2/core/eigen.hpp>


struct ReprojectionError 
{
    cv::Point2f observed;
    cv::Mat K;

    ReprojectionError(cv::Point2f obs, cv::Mat K) : observed(obs), K(K) {}

    template <typename T>
    bool operator()(const T* const rotation, const T* const translation, const T* const point3D, T* residuals) const {
        T R[9];
        ceres::QuaternionToRotation(rotation, R);

        T p_h[4] = { point3D[0], point3D[1], point3D[2], T(1.0) };

        T p_rot[3];
        p_rot[0] = R[0] * p_h[0] + R[1] * p_h[1] + R[2] * p_h[2];
        p_rot[1] = R[3] * p_h[0] + R[4] * p_h[1] + R[5] * p_h[2];
        p_rot[2] = R[6] * p_h[0] + R[7] * p_h[1] + R[8] * p_h[2];

        T p_cam[3];
        p_cam[0] = p_rot[0] + translation[0];
        p_cam[1] = p_rot[1] + translation[1];
        p_cam[2] = p_rot[2] + translation[2];

        T fx = T(K.at<double>(0, 0));
        T fy = T(K.at<double>(1, 1));
        T cx = T(K.at<double>(0, 2));
        T cy = T(K.at<double>(1, 2));

        T u = fx * (p_cam[0] / p_cam[2]) + cx;
        T v = fy * (p_cam[1] / p_cam[2]) + cy;

        residuals[0] = u - T(observed.x);
        residuals[1] = v - T(observed.y);
        return true;
    }
};
    
void bundle_adjustment(std::vector<Keyframe>& keyframes, cv::Mat& K) 
{
    std::cout << "Running Full Bundle Adjustment..." << std::endl;
    ceres::Problem problem;

    for (auto& keyframe : keyframes) 
    {
        Eigen::Matrix3d R_cv;
        cv::cv2eigen(keyframe.rotation, R_cv);

        Eigen::Quaterniond quaternion;

        ceres::RotationMatrixToQuaternion(R_cv.data(), quaternion.coeffs().data());

        double rotation[4] = {quaternion.x(), quaternion.y(), quaternion.z(), quaternion.w()};
        double* translation = keyframe.translation.ptr<double>();
        for (size_t i = 0; i < keyframe.points2f.size(); i++) 
        {
            cv::Point3f point3D_f = keyframe.points3D[i].point3D;
            double point3D[3] = {static_cast<double>(point3D_f.x), 
                                static_cast<double>(point3D_f.y), 
                                static_cast<double>(point3D_f.z)};
            
            auto cost_function = new ceres::AutoDiffCostFunction<ReprojectionError, 2, 4, 3, 3>(
                new ReprojectionError(keyframe.points2f[i], K)
            );

           problem.AddResidualBlock(cost_function, nullptr, rotation, translation, point3D);
        }

    }

    // **Configure Ceres Solver Options**
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.minimizer_progress_to_stdout = true;
    options.max_num_iterations = 100;

    // **Solve the problem**
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::cout << summary.FullReport() << std::endl;
}
