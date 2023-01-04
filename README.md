# expfracML

These are the files for the project and a paper "Deep Learning to Predict Permeability Through Rough Fractures for Multi-Physics Modelling". 

Data is missing here as it is large ~3GB, please email me to recive the data until I sort out how to make it publicly availible. 

Author: Aidan Kubeyev
E-mail: 77777aidan@gmail.com


Abstract:
Numerical modelling of fractures or other complex structural discontinuities is computationally demanding. Typically, it involves solving the multi-scale and multi-physics phenomena, simulating combined solid, fluid, thermal and chemical processes. To address long computation times limiting assumptions, simplifications or conceptualizations have to be made. In multi-physics modelling, it is desirable to predict certain parameters quickly without making simplifications. In this paper, a data-driven deep learning modelling approach is presented that learns to predict physical permeability parameters through a fracture with complicated geometry based on a digital image.
Besides, this study investigates the ability of the Convolutional Neural Network (CNN), a type of deep learning, to be used in a regression framework, to predict a continuous value. Images of fractures were generated using a digitalised rough fracture surface from a subsurface rock. For the training and testing, permeability was calculated using the Stokes equation and the Finite Volume discretisation. Two cases were investigated: when the fracture fluid velocity field is provided for the CNN to train, and a more difficult case when the velocity field is not provided. It was found that deep learning can learn from cross-sectional images of fractures to predict permeability without fluid velocity provided, achieving high accuracy. It was also found that the CNN generalises well, accurately predicting the permeability of significantly different fractures, parametrised by roughness. Overall, the approach can reduce the computation time during multi-physics modelling and can be used to predict continuous physical permeability values of a fracture with a complex surface from an image. }
