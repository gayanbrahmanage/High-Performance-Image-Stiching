#ifndef IMAGE_FILE_LISTER_HPP
#define IMAGE_FILE_LISTER_HPP

#include <vector>
#include <string>
#include <filesystem>
#include <algorithm>
#include <iostream>

namespace fs = std::filesystem;

inline std::vector<std::string> listJPGFiles(const std::string& folder) {
    std::vector<std::string> fileNames;

    if (!fs::exists(folder) || !fs::is_directory(folder)) {
        std::cerr << "Invalid folder: " << folder << std::endl;
        return fileNames;
    }

    // Collect all .jpg files
    for (const auto& entry : fs::directory_iterator(folder)) {
        if (entry.is_regular_file()) {
            auto path = entry.path();
            if (path.extension() == ".jpg") {
                fileNames.push_back(path.string());
            }
        }
    }

    // Sort by filename (e.g., 1.jpg, 2.jpg, ...)
    std::sort(fileNames.begin(), fileNames.end());

    return fileNames;
}

#endif // IMAGE_FILE_LISTER_HPP
