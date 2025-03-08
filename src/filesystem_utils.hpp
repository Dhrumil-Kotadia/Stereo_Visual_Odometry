#include <filesystem>

directory_datatype create_directories(std::string directory_path = "..") 
{
    directory_datatype directories;
    directories.read_directory_path_1 = directory_path + "/data/images_left_7/";
    directories.read_directory_path_2 = directory_path + "/data/images_right_7/";
    directories.save_directory_path = directory_path + "/debug/images/";
    directories.detections_directory_path = directory_path + "/debug/images/detections/";
    directories.matches_directory_path = directory_path + "/debug/images/matches/";

    std::filesystem::create_directory(directories.save_directory_path);
    std::filesystem::create_directory(directories.detections_directory_path);
    std::filesystem::create_directory(directories.matches_directory_path);

    return directories;
}

cv::Mat read_image(const std::string &path) 
{
    cv::Mat img = cv::imread(path, cv::IMREAD_GRAYSCALE);
    if (img.empty()) {
        std::cerr << "Error: Image not found at " << path << std::endl;
        exit(1);
    }
    return img;
}

void save_image(const std::string path, const cv::Mat img) 
{
    cv::imwrite(path, img);
}

std::vector<std::string> read_directory(const std::string &directory_path) 
    {
        std::vector<std::string> image_paths;
        for (const auto &entry : std::filesystem::directory_iterator(directory_path)) {
            if (entry.path().extension() == ".png") {
                image_paths.push_back(entry.path());
            }
        }
        return image_paths;
    }
