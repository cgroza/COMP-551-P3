# README - COMP 551 Project 3
Cristian Groza, Alicec Jiang, Rubia Albuquerque

## Used libraries

- pytorch
- numpy

## Replicating results

### Generating submissions ###

- Edit script and set _submission_ flag to True.
- Run experiments.py
- Script will train 20 models, and bag them for prediction. Models will be saved
  to disk.
- Generated _submission.csv_ contains the testing set predictions.

### Generating validation results
- Edit script and set _validation_ flag to True.
- Run experiments.py
- Script will run the 20 models on the validation training set.
- Will output validation accuracy.
