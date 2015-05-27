cmake_minimum_required(VERSION 2.8.3)
project(eye_tracker)

find_package( OpenCV REQUIRED )

set( OBJ_DET_SRC face_detect/objectDetection2.cpp 
		 face_detect/findEyeCenter.cpp 
		 face_detect/findEyeCorner.cpp
		 face_detect/helpers.cpp 
)
# save all binaries to bin folder in top level of project
set( EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/test ) 
#set( EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin ) 

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
add_executable(obj_detect ${OBJ_DET_SRC})

## Specify libraries to link a library or executable target against
target_link_libraries(eye_tracker 
  ${OpenCV_LIBS}
)
target_link_libraries(obj_detect
  ${OpenCV_LIBS}
)
