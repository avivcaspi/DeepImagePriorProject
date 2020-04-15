This is readme file that explains the structure of the project code,
and how to run and reproduce all results.

Project Structure:
src/models/ - include all models implementation - Unet and it`s blocks and adversarial 
model and training.

src/data/ - include all data for the experiments - images and masks for all applications.

src/checkpoints/ - include some checkpoints to run experiments without training

src/results/ - include all result images from all notebooks and applications

src/utils.py and src/input_gen.py - utils files used for loading and manipulating images
and generating the input for each application

src/*.ipynb - notebooks that present all application and training code

report.pdf - The report of the project


Reproducing the results:
In order to reproduce all results, all you need to do is open one of the notebooks
(one notebook for each application) and run the notebook from start to finish
All the notebook will start by loading the images, regular training of the model and 
presenting the results, than training with input optimization and presenting the results,
and lastly adversarial training and showing the results.
Please note that some of the training might take a long time, therefore we ran each notebook
and uploaded all result to each notebook, so you can look at the notebooks results without
running them.

If you want to avoid trainig all the notebooks but still observe the results you can got to 
out git repository https://github.com/avivcaspi/DeepImagePriorProject
and download checkpoints with final of the notebook you want and than run the notebook again.
(We didn`t add the checkpoints to the submission because each checkpoint weight 30Mb)