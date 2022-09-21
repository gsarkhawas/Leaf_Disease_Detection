## Leaf_Disease_Detection

Early diagnosis and accurate identification of leaf diseases can control the spread of infection and
ensure the healthy development. Manual inspection or description about the quality of plant 
such as normality or health of leaves are not accurate and furthermore cannot be taken as 
decision factor for productivity improvement of crops. Automatic systems for
identification and recognition of plant diseases are helpful to monitor large fields and identify plant
diseases automatically based on symptoms that are visible on leaves of plants.

This project focuses on utilizing deep learning algorithm with `keras` on potato images collected from `crowdAI site
for plant village classification challenges`. 

Folder strcture:

- `ModelTrain.py` script represents the training model
- `ModelTest.py` script represents the testing module used for prediction
- `upload.py` script represents the prediction code for an unseen test leaf image which can be uploaded
on a localhost using flask api
- The saved model and its weights are stored in `final_model.json` and `final_model_weights.h5`
