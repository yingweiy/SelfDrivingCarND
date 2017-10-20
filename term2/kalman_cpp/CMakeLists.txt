project(KalmanFilter)
set (SOURCE_FILES main.cpp kalman_filter.cpp kalman_filter.h tracking.cpp tracking.h measurement_package.h)
include_directories(./Eigen)
add_executable(laser ${SOURCE_FILES})
