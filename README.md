# Digit Identification

## Overview

This is a hobby project to improve on my skills using a combination of ML techniques and application deployment. While the ML aspects of this problem are fairly simple, it is good practice with training and deploying ML models for applications in a containterized environment.

## ML Model
- In the python notebook, i train a CNN on images of drawn digits from the MNIST dataset.
- ML modelling is done using PyTorch.
- The model takes a 28x28 pixel image as input and the forward pass decides on which digit it is most likely to be.
- 10,000 images are in the training set.

## Application
- The application is deployed as a composition of these 3 docker containers: app, model-service, and postgreSQL DB.
- The main app is built using streamlit and exposed through port 8501, with access to the other two containers.

