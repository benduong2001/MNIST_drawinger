# MNIST_drawinger
Number classification with drawing interface using tkinter, the MNIST dataset, keras's convolutional neural networks.
- Capable of recognizing numbers, meaning multiple digits, in one picture. 
- currently uses DBSCAN algorithm for image segmentation.
## How to use:
- Wait until the tkinter canvas window opens
- Draw some numbers. To erase, draw while holding right click
- Click Save
- The code will replace any file with the name "conv (temp_mnist).jpg" in the current directory folder. While this is highly unlikely, please make sure you don't have any files with that exact name in the directory folder.
- Click test, and the code will guess the number written.
