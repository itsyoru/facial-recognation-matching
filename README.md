Using OpenCV & DeepFace for live facial recognation 

Upload your own reference img in the project folder named 'reference.jpg'

If you have more than 1 camera's you can edit the value in this line to 1, 2 etc. -->  cap = cv2.VideoCapture(0)  

To run, open terminal and run python3 main.py

The program will check if the live face being captured is a match or not with the reference image, while also providing an emotion + age prediction 

Press Q on your keyboard to close the program
