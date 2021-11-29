# PPE Helmet detection Project 
This study aims to develop a framework to sense in real-time the safety compliance of workers with respect to PPE. 
The final model can detect whether the human subject is wearing helmet or not. 


## Description of the project
Transfer learning is used in this project, YOLO V2 is used as the base model, one convolutional layer and the output layer were modified according to current project. 


i.	Neural network architecture
![Alt text](PPE-Project/asset/network.PNG?raw=true "Title")



## Data Set Sources

The source of the dataset is from Kaggle, [here](https://www.kaggle.com/agrigorev/clothing-dataset-full) is the link to the dataset. 

###	Model Training
Dataset with annotation in Xml format is used to train the neural network. 
###	Testing
![Alt text](PPE-Project/asset/example.PNG?raw=true "Title")

###	Future Development
The model can differentiate human subject wearing helmet vs not wearing helmet well. However, the model is not very accurate yet, more tuning and improvement is needed to improve the accuracy. 


###	List of Group members
- Jiong Jiet: cjj8168@gmail.com
- Aslam: ahmadaslam3838@gmail.com 
 
