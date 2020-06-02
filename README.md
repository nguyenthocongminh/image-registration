# image-registration

#### Introduce
This project was created for the purpose of an image processing application to find the location of a partial photograph of a car in that car's overall photo

### REQUIREMENTS
This project is written in python with version 3.7.6
Required extensions:
- imutils
- matplotlib
- numpy
- opencv-contrib-python (version: 3.4.2.16)
- opencv-python 		(version: 3.4.2.16)

### Setup project
1. Install python version 3.7.6\
	[Download here](https://www.python.org/downloads/release/python-376/)
2. Install the required ext has been written in `requirement.txt` file
	- Open a Command Prompt (cmd) or PowerShell on the folder of project then typing this
	```
	pip install -r requirements.txt
	```
	
### How to use
1. If you just have two image (one is a part of car and one is overall), you can make a python file with
	```
	from imageRegistration import image_registration as ir
	
	predict, image = ir(path_to_part_car_image, path_to_overall_car_image, algorithm)
	```
	With param
	- `path_to_part_car_image` is the path to the file part car image
	- `path_to_overall_car_image` is the path to the file overall car image
	- `algorithm` is the algorithm find feature point (1 is sift, 2 is surf, 3 is brisk)
	
	And result
	- `predict` is tell you if it can find the part car image in the overall car image
	- `image` is the picture that show you position of the part car image in the overall car image
2. If you want to compare the result between algorithms you can go to file `test.py`, edit `root_test_path` to the path of your root set test and run, you can see result of each algorithm (example count pass, count fail, ...)