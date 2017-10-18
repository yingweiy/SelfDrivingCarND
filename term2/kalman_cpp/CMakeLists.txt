project(KalmanFilter)
set (SOURCE_FILES kalman_main_single_file.cpp)
include_directories(./Eigen)
add_executable(kalman1 ${SOURCE_FILES})
