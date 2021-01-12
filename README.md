# seal | see-all
*seal* is an application made to detect object distances for drivers with some kind of visual impairment or in unusual conditions such as rainy nights.

It was made using Python, Numpy and OpenCV to detect objects or cars that could possibly be closer than they appear. People with visual impairment, such as myopia,
might have trouble to judge an object's distance from their car, especially at night or in rainy days. 

*seal* uses the FAST detection algorithm for keypoint detection, which points out every nuance in an image that might be a corner of some object. 
From there, it maps every point that might be inside or closer to the area of the camera that sees what the car "sees", like lanes in a highway
(a triangle from both lower corners to the middle of an image, forming a small pyramid). For precision, the user can run the program by passing in an argument being `roof` or `panel`, depending on the physical position of the camera. 
The program will adjust itself based on the argument and set the camera axis as accordingly as possible.

*seal* **can** detect lanes when in a highway, although it does not depend on this kind of detection to check the distance between the object and the car.

This project **is not yet ready for production, as it lacks polishing and precision when checking for objects at high speed and/or different lighting conditions**.
If you have any experience on Computer Vision and feel like contributing to this project, I will be more than happy to help.
