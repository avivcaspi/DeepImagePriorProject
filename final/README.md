# DeepImagePriorProject

Project Structure:

src/models/ - includes all models implementation - Unet and its blocks, and adversarial 
model and training.

src/data/ - includes all data for the experiments - images and masks for all applications.

src/checkpoints/ - include some checkpoints (trained models) to run 
experiments without training.

src/results/ - includes all result images from all notebooks and applications.

src/utils.py and src/input_gen.py - utility files used for loading and manipulating images
and generating the input for each application.

src/*.ipynb - notebooks that present all application and training code.

report.pdf - The report of the project


Reproducing the results:
In order to reproduce all of the results, you first need to install an anaconda environment
from the file src/environment.yml , so that you will have all the required packages.
After that, all you need to do is open one of the notebooks (one notebook for each 
application) and run the notebook from start to finish.
Every notebook will start by loading the images, regular training of the model and 
presenting the results, than training with input optimization and presenting the results,
and lastly adversarial training and showing the results.
Please note that some of the training might take a long time, so we ran each notebook
and uploaded all result for each notebook, so you can look at the notebook results without
running them.

If you want to avoid training all the notebooks but still observe the results you can go to 
our git repository at https://github.com/avivcaspi/DeepImagePriorProject and download 
checkpoints with the final version of the notebook you want and then run the notebook again.
(We didn`t add the checkpoints to the submission because each checkpoint weighs around 30Mb)