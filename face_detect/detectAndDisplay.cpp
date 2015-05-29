
/**
 * @function detectAndDisplay
 */

//--(!) figure out how to make this faster!
// a: nuke current ubuntu 12.04 and install ubuntu 12.04LTS
// b: only pass mats by reference

//-- instead of frame, pass face_search_space
int detectAndDisplay( Mat &frame )
{
    int numFaces = 0;

    cvtColor( frame, frame_gray, COLOR_BGR2GRAY );

    equalizeHist( frame_gray, frame_gray ); // maybe remove this for speed if unnecessary for quality
#ifdef DEBUG
    clock_t start;
    double dur;

    start=clock();
    face_cascade.detectMultiScale( frame_gray, faces, 1.1, 2, 0, Size(150, 150),Size(300,300) );
    dur = ( clock()-start )*1000./(double)CLOCKS_PER_SEC;
    cout << "Detection Durations\n";
    cout.precision(numeric_limits<double>::digits10);
    cout << "Face Cascade: " << fixed << dur << "ms\n";
#else
    //-- Detect faces
    face_cascade.detectMultiScale( frame_gray, faces, 1.1, 2, 0, Size(150, 150),Size(300,300) );
#endif

    for( size_t i = 0; i < faces.size(); i++ )
    {
        //-- Draw the face
 //       rectangle( frame, faces[i],  Scalar( 255, 0, 0 ), 2, 8, 0 );
        cout<< "face rect: " << faces[i] << endl;
        numFaces++;
        faceROI = frame_gray( faces[i] );
#ifdef DEBUG
        start = clock();
        eyes_cascade.detectMultiScale( faceROI, eyes, 1.1, 2, 0 |CV_HAAR_SCALE_IMAGE, Size(50,50),Size(80,80) );
        dur = ( clock()-start )*1000./(double)CLOCKS_PER_SEC;
        cout << "Eye Cascade: " << fixed << dur << "ms\t" << eyes.size() << "eyes found\n";
#else
        //-- In each face, detect eyes
        eyes_cascade.detectMultiScale( faceROI, eyes, 1.1, 2, 0 |CV_HAAR_SCALE_IMAGE, Size(50,50),Size(80,80) );
#endif
        if( eyes.size() == 2)
        {
         for( size_t j = 0; j < eyes.size(); j++ )
         {
           //-- Draw the eyes
            eye.x = eyes[j].x + faces[i].x;
            eye.y = eyes[j].y + faces[i].y;
            eye.width = eyes[j].width;
            eye.height = eyes[j].height;
//            rectangle( frame, eye, Scalar( 255, 0, 255 ), 2, 8, 0 );


#ifdef DEBUG
            start = clock();
            Point eye_center = findEyeCenter(faceROI, eyes[j], "Debug Window");
            dur = ( clock()-start )*1000./(double)CLOCKS_PER_SEC;
            cout << "Eye Center: " << fixed << dur << "ms\n";
#else
            // Find Eye Center
//            Point eye_center( eyes[j].width/2, eyes[j].height/2 );  // middle of eye rectangle
            Point eye_center = findEyeCenter(faceROI, eyes[j], "Debug Window");
#endif

            // if eye.x > face_width/2 --> right eye --> use kal1, else use kal2


            // take average of eye_center:eye width for both eyes
            if(j==0){
                x_frac = (double)eye_center.x / (double)eye.width;
                y_frac = (double)eye_center.y / (double)eye.height;
            } else {
                x_frac = 0.5 * (x_frac + ((double)eye_center.x / (double)eye.width));
                y_frac = 0.5 * (y_frac + ((double)eye_center.y / (double)eye.height));
            }

            // Shift relative to eye
            eye_center.x += faces[i].x + eyes[j].x;
            eye_center.y += faces[i].y + eyes[j].y;

            // Draw Eye Center
            crosshair1.x = eye_center.x;
            crosshair1.y = eye_center.y-5;
            crosshair2.x = eye_center.x;
            crosshair2.y = eye_center.y+5;
            line( frame, crosshair1, crosshair2, Scalar( 0, 255, 0 ), 1, 8, 0 );
            crosshair1.x = eye_center.x-5;
            crosshair1.y = eye_center.y;
            crosshair2.x = eye_center.x+5;
            crosshair2.y = eye_center.y;
            line( frame, crosshair1, crosshair2, Scalar( 0, 255, 0 ), 1, 8, 0 );
          }
        }
    }
    cout <<"\n";

   //-- Show what you got
   imshow( window_name, frame );


#ifdef GAME_PLAY
   // these are the ratios of point coords to eye rect dims (x_frac and y_frac)
   // Point: (H, W)
   // TL: (.44, .54) -> (1,0)
   // TR: (.56, .54) -> (1,1)
   // BL: (.44, .46) -> (0,0)
   // BR: (.56, .46) -> (0,1)

   x_frac = 8.33*x_frac - 3.67;
   x_frac = min(max(x_frac, 0.0), 1.0); // boundary condition if x_frac is neg
   y_frac = 12.5*y_frac - 5.75;
   y_frac = min(max(y_frac, 0.0), 1.0);

   // Show where you're looking
   game_frame = Mat::zeros(height_screen, width_screen, CV_8UC3);
   gaze.x = x_frac * width_screen;
   gaze.y = y_frac * height_screen;
   circle(game_frame, gaze, 10, Scalar(255, 255, 255), 20);
   imshow(game_window, game_frame);
   printf("x = %f, y = %f\n", x_frac, y_frac);
#endif

   return numFaces;


}
