#ifndef GLOBALS_H
#define GLOBALS_H

// GUI Dimensions
const std::string game_window = "Eye Tracking Game";
const int width_screen = 1590;
const int height_screen = 1100;

const std::string face_cascade_path = "/home/ee443/FinalProjectEE443/eye_tracker/src/face_detect/data/lbpcascades/";
const std::string eye_cascade_path = "/home/ee443/FinalProjectEE443/eye_tracker/src/face_detect/data/haarcascades/";
//std::string eye_cascade_path = "/home/ee443/FinalProjectEE443/eye_tracker/src/face_detect/data/haarcascades_cuda/";

const std::string face_cascade_name = "lbpcascade_frontalface.xml";
const std::string eyes_cascade_name = "haarcascade_eye_tree_eyeglasses.xml";
//std::string eyes_cascade_name = "haarcascade_eye.xml";
//std::string eyes_cascade_name = "haarcascade_lefteye_2splits.xml";
//std::string eyes_cascade_name = "haarcascade_righteye_2splits.xml";

#endif // GLOBALS_H
