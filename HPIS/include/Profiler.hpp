#ifndef PROFILER_HPP
#define PROFILER_HPP

#include <chrono>
#include <iostream>
#include <fstream>
#include <string>

class Profiler {
public:
    // Start the timer
    void start() {
        start_time = std::chrono::high_resolution_clock::now();
        start_memory = getMemoryUsageKB();
    }

    // End the timer and print results
    void end(const std::string& processName = "") {
        auto end_time = std::chrono::high_resolution_clock::now();
        auto end_memory = getMemoryUsageKB();

        double duration_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
        long memory_diff = end_memory - start_memory;

        std::cout << "===== Profiler Results =====" << std::endl;
        if (!processName.empty()) {
            std::cout << "Process: " << processName << std::endl;
        }
        std::cout << "Elapsed time: " << duration_ms << " ms" << std::endl;
        std::cout << "Memory change: " << memory_diff << " KB" << std::endl;
        std::cout << "============================" << std::endl;
    }

private:
    std::chrono::high_resolution_clock::time_point start_time;
    long start_memory;

    // Get current memory usage in KB (Linux)
    long getMemoryUsageKB() {
        std::ifstream stat_file("/proc/self/status");
        std::string line;
        while (std::getline(stat_file, line)) {
            if (line.substr(0, 6) == "VmRSS:") {
                std::string value = line.substr(6);
                return std::stol(value); // KB
            }
        }
        return 0;
    }
};

#endif // PROFILER_HPP
