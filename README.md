# TG2

Files used for my undergraduate thesis.

## Setup

Before running any scripts, check if all requirementes listed on `requirements.txt` are met. You also need to have ROS installed. Download the files at <https://drive.google.com/open?id=1NQWUPifuFnISEpX4DY528gD1dfKHTAjH> and unpack them inside `bags`, `data`, `models` and `results`. Unnest the folders if necessary.

## Folders

Inside `bags` are the captured bags used for the experiments. Inside `data` the data from the bags is already extracted and split into train, validation and test sets. The datasets created are `predio` and `oito`. Inside `models` are all the models resulting from the experiments. The folders are the same as the folders in `data`. Lastly, inside `results` there is data collected from the validation with inexperienced users.

## Execution

To extract data from `.bag` files, you can use `split_bags.sh` to create train, validation and test sets, the use `extract_bags.sh` to extract the data from the `.bag` files to `.csv` files. `extract_bags.sh` uses `extract_data.py`. If you want to view the data before training, you can use `view.py`.

To train, first adjust `config.py` to your needs. Then run `cv_train.py` to train the various networks architectures listed on the configuration file with cross-validation, and run `cv_compare.py` to list the architectures in order of RMSE. You can test the architectures and plot graphs with `cv_test.py`. All models trained with cross-validation are saved under the folder `models/cross-validation`.

Once you picked one or more architectures, you need to now train once again but using the train, validation and tests splits created earlier. So once again, adjust `config.py` to your needs, then run `train.py` to train the various networks architectures listed on the configuration file, and run `compare.py` to list the architectures in order of RMSE. In this case, all models are saved under the folder `models`. You can then run `test.py` to check the performance of a specific model and plot graphs.

The `send_*.py` files are related to Intel DevCloud. They are used to submit jobs to the servers, and they use the programs mentioned above for cross-validation and training.

Inside `ros` is the ROS packages used to implement the correction algorithm. The node is implemented in `correct.py`. The interface with the controller can be started with `joy_remaped.launch`. The remapping is nedded for the commands to pass through the node before being sent to the robot.

To extract the results from the bags files, you can use `split_results.sh` to split the bag into the methods tested and then use `extract_results.sh` to extract the results from the generated bag. `split_results.sh` uses `split_results.py` and `extract_results.sh` uses `extract_results.py`. You can then plot graphs with `pie.py` and `analyse.py`.
