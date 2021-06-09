# Robot-Failure-NN
This code is about robot execution failures using neural networks classification. Training data included 550 samples with 6 features. Features were measured forces and torques of a robot after failure detection in x, y and z coordinates. Neural network is trained with training data and tested with 10 samples. Neural network predicted output with 80 % accuracy. Algorithm is working but accuracy can be considered low. More careful observation of test data reveals that one measurement is not 15 samples with 6 features but 1 sample with 90 features. By using training data this way, accuracy could have been much higher. Complete python code and edited data is provided in this repository.

Neural network code is based on example by Dr. Michael J. Garbade. Example can be found in: https://www.kdnuggets.com/2018/10/simple-neural-network-python.html

Free dataset used in this project was downloaded from University of California Irvine Machine Learning Re-pository. https://archive.ics.uci.edu/ml/datasets/Robot+Execution+Failures
