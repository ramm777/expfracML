# expfracML

These are the files for the project "Deep Learning for Identification of Permeability in Rough Fractures". 

Data is abscient as it is large ~3GB, please email me to recive the data until I sort out how to make it publicly availible. 

Author: Aidan Kubeyev
E-mail: 77777aidan@gmail.com

Abstract:
Rock fractures are complex structural discontinuities, where modelling involves solving the multi-scale and multi-physics phenomena. 
Typically, the simulation of combined solid, fluid and chemical processes is computationally demanding. In this paper, I present a 
data-driven deep learning modelling approach that inputs a digital image of a rough fracture with a complicated surface and predicts
the permeability through it. Besides, I investigate the ability of the Convolutional Neural Network (CNN), a type of deep learning, 
to be used in a regression framework, predicting a continuous rough fracture permeability. Fracture images were created using a 
digitalized rough fracture surface from a mudrock core. For the machine learning training, fracture permeability was calculated using
the Stokes equation and the Finite Volume discretisation. I investigate two scenarios: when the fracture fluid velocity field is provided 
for the CNN to train, and a more difficult scenario when the velocity field is not provided. I find that deep learning can learn from 
cross-sectional images of fractures to predict fracture permeability without fluid velocity, achieving an accuracy of 97%. Overall, the 
approach can improve the multi-physics modelling computational time, and deep learning can be used for regression in geosciences. 
