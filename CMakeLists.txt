cmake_minimum_required(VERSION 2.8.3)
project(eye_tracker)

find_package( OpenCV REQUIRED )

## Build ##
include_directories(
  ${OpenCV_INCLUDE_LIBRARIES}
)

#INCLUDE( ${QT_USE_FILE} )
#ADD_DEFINITIONS(${QT_DEFINITIONS}
# This macro invokes moc: processes headers and stores src files in
# _HDR_MOC 
#QT4_WRAP_CPP( tracker_HDR_MOC ${tracker_HDR} )

## Declare a cpp executable
add_executable(eye_tracker eye_tracker.cpp)

## Specify libraries to link a library or executable target against
target_link_libraries(eye_tracker
  ${OpenCV_LIBS}
)
