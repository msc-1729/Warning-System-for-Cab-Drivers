<h2 align = "center"> Warning System for Cab Drivers </h2>

### Introduction
This is a Python multi-threaded program that analyses the surroundings of the driver such as recognizing the traffic signs, detecting the objects in front of the driver, and warning him accordingly through speech output. 

### Why Warning a Driver?
The massive production and also the use of vehicles is not only degrading the environment by increasing the pollution but also this is  resulting in increasing the risk and probability of fatality and destruction due to vehicle accidents. In these accidents most of the cases are mainly due to the reckless behaviour of the vehicle drivers which can have many reasons like restlessness for the driver etc. Road accidents have become very common nowadays. As more and more people are buying automobiles, the incidences of road accidents are just increasing day by day. Furthermore, people have also become more careless now. Not many people follow the traffic rules. Especially in big cities, there are various modes of transports. Moreover, the roads are becoming narrower and the cities have become more populated. Hence, this proves that there is need for a system that warns the driver and assists him in reaching the destination safely. To prevent all this fatality and disasters that happen in the society the drivers should get some kind of assistance that guide them or suggest them whilst driving. It's never a feasible or efficient idea to expect a person to be with you when you are driving to assist you. Hence a system can be made that can provide assistance to the driver. The system should be able to analyse the situation in the surrounding areas of the vehicle and should be able to decide whether there is a crowd in that area. The system should be able to take the input from the attached input camera device and should process the input to produce decisions that it makes by analysing the input in runtime. 

### Solution
A software system that monitors the surroundings while driving, in real time by capturing the video of the surroundings through camera input and creates awareness about the situation to the driver by analysing the input it gets through the camera using the neural networking algorithms that have been developed and assists the user through relevant messages and appropriate signals that have been produced through decision made by the analysis.

### Use Case Diagram
In order to depict the proper functionality of the system and to have an idea of what tasks have to be acheived inorder to suffice all the requirements for creating this warning system.

<img src = "https://github.com/msc-1729/Warning-System-for-Cab-Drivers/blob/main/assets/Use%20Case%20Diagram%20of%20Warning%20System%20for%20Drivers.png" />

### Low Level Architecture

<img src = "https://github.com/msc-1729/Warning-System-for-Cab-Drivers/blob/main/assets/Low%20Level%20Architecture.PNG" />

### Implementation
●	CROWD INTENSITY DETECTION:
    A pre-trained model from a research paper that uses a cascading algorithm to detect the persons is used. This pre trained model takes in the input image as an array        and returns the coordinates of the region where the person is present and also the probability of how accurate the model is about that object being a person.
    Using the coordinates a boundary box is drawn around the person if the probability that has been returned from the model is greater than the threshold probability and      the count of the persons is incremented. If the count of the persons is greater than a number defined, then the message 
    ‘Crowdy Area’ is produced and is given as an input to the text to speech converting engine to produce the output.

●	TRAFFIC SIGN DETECTION:
   The dataset required for training the model has been taken from the kaggle website. This data set is augmented and processed using image processing techniques. The         input image is converted to grayscale image and finally the pixel range is normalised to a range of 0 - 1. These images are used for the training of the model. The         convolution neural network has 64 2DCNN layers which have an activation function of relu for all the inner layers and each layer has an average pooling. All these          layers are brought down to a dense layer of 43 nodes with an activation function softmax as there are 43 different classes as per the dataset available. Optimizer          Adadelta is used  while training the model. This model after training could achieve a validation accuracy of 96%. This model is saved into a pickle file which is           further used in the process where it gets an input from the frame that has been taken as an input and the traffic sign is recognised and the accuracy of how probable       that the image is a traffic sign is returned. Based on the accuracy being greater than the threshold value the message is produced which has the information about the      traffic sign. This is given as an input to the text to speech converter and an output is generated accordingly.

The project is started off with importing the libraries that are required for loading the dataset and pickle files, implementing new threads for each task, and for converting text into speech.

<img src = "https://github.com/msc-1729/Warning-System-for-Cab-Drivers/blob/main/assets/Libraries%20used%20in%20the%20project.png"/>

<img src = "https://github.com/msc-1729/Warning-System-for-Cab-Drivers/blob/main/assets/Loading%20data%20using%20Pickle%20loader.png"/>

Our Project is categorized as:
●	Converting the loaded images into gray scale images and then normalising them.
●	Creating a two dimensional CNN.
●	Compiling the CNN with the normalised data.

These three steps can be visualized as: 

<img src="https://github.com/msc-1729/Warning-System-for-Cab-Drivers/blob/main/assets/Steps%20of%20execution.png"/>

The sequential API allows one to create models layer-by-layer for most problems. It is limited in that it does not allow you to create models that share layers or have multiple inputs or outputs. As we have a single source for inputs, we prefer sequential API over Functional API.
The kernel size of (5 X 5) which is 2-tuple with the width and height of the 2D Convolution window is used. RELU is taken as the activation function as it is considered to be as the best for hidden layers of a CNN. The input size contains the dimensional information of the input images. Here it is specified as (32,32,1) as per the dimensions of the input image. After applying a nonlinear activation function like RELU to the feature maps output by the CNN, adding a pooling layer to the CNN helps in down sample the feature maps by summarising the presence of features in patches by using strides or filters. Average Pooling is used in this case to down sample the features.
Optimiser Adadelta is used and as the data is sparse with 43 output classes we use the loss function as sparse categorical cross entropy with accuracy as the metric for measurement.

History variable contains the values of accuracy and loss for training and validating the data during each epoch and these can be used to plot the graphs for visual assistance to evaluate the results produced.
<img src = "https://github.com/msc-1729/Warning-System-for-Cab-Drivers/blob/main/assets/Training%20vs%20valid.png" />
<img src = "https://github.com/msc-1729/Warning-System-for-Cab-Drivers/blob/main/assets/Training%20vs%20validation%20loss.png" />

●	INTEGRATING THE PROCESSES
After getting the input image from the input camera, the processes that take this frame as input have to work simultaneously. To make them work simultaneously the procedure of multi threading is chosen over multiprocessing because multithreading uses a single memory unit for all the threads and also we have only a single image on which all these processes have to work. The first thread is used for processing the image ,detecting the crowd intensity and the traffic sign present in the image. The second thread is used to produce the audio output for the crowd intensity. The third thread is used to produce the audio output for the traffic sign detected. 
●	CONVERTING TEXT TO SPEECH:
To convert the text to speech pyttsx3 library is used. This has a predefined function that converts the text to speech offline without any use of the API’s that use the internet.

### Obtained Results

<img src= "https://github.com/msc-1729/Warning-System-for-Cab-Drivers/blob/main/assets/Result%20of%20traffic%20recognition.png"/>
<img src="https://github.com/msc-1729/Warning-System-for-Cab-Drivers/blob/main/assets/Recognizing%20speed%20board.png"/>
<img src="https://github.com/msc-1729/Warning-System-for-Cab-Drivers/blob/main/assets/Recognizing%20people.png"/>









