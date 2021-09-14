Each model's directory will contain these files:

| First Header  | Second Header |
| ------------- | ------------- |
| download.txt  | A text file that contains a link from where you can download the trained model |
| train.py  | Training script, that trains the model found in `training_output` sub-folder for a given number of epochs |
| test.py  | Testing script, that runs the model found in `training_output` in inference mode for 6 randomly preselected images (3 from training, 3 from evaluation datasets) and displays all predictions on each image in a pop-up window |
| eval.py  | Testing script, that runs the model found in `training_output` in inference mode for all images found in dataset's `val` folder and `val.json`. Evaluation is performed through Detectron's `COCOEvaluator` |