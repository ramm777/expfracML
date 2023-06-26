# Enhancing Multi-Physics Modelling with Deep Learning: Predicting Permeability through Structural Discontinuities 
## Here, I present a method of utilizing deep learning to enhance physics-based solid-fluid mechanics modelling by replacing one part of it, permeability through complex geometries. 

# Abstract
Numerical modelling of complex structural discontinuities such as fractures is computationally demanding. Typically, it involves solving the multi-scale and multi-physics phenomena, simulating combined solid, fluid, thermal and chemical processes. To address long computation times limiting assumptions, simplifications or conceptualizations have to be made. In multi-physics modelling, it is desirable to predict certain parameters quickly without making simplifications. In this paper, a data-driven deep learning modelling approach is presented that learns to predict physical permeability parameters through a fracture with complicated geometry based on a digital image. Besides, this study investigates the ability of the Convolutional Neural Network (CNN), a type of deep learning, to be used in a regression framework, to predict a continuous value. Images of fractures were generated using a digitalised rough fracture surface from a subsurface rock. For the training and testing, permeability was calculated using the Stokes equation and the Finite Volume discretisation. Two cases were investigated: when the fracture fluid velocity field is provided for the CNN to train, and a more difficult case when the velocity field is not provided. It was found that deep learning can learn from cross-sectional images of fractures to predict permeability without fluid velocity provided, achieving high accuracy. It was also found that the CNN generalises well, accurately predicting the permeability of significantly different fractures, parametrised by roughness. Overall, the approach can reduce the computation time during multi-physics modelling and can be used to predict continuous physical permeability values of a fracture with a complex surface from an image. 
# Requirements

- Python and dependencies --> please see the requirements.txt file for the full requirements of each module. 
- Matlab --> 2017 or above, MRST-2017a or above (Matlab Reservoir Simulation Toolbox https://www.sintef.no/projectweb/mrst/). 


# Installation 

- Matlab --> once installed MRST, please place the Matlab code to the "vem" module inside Mrst. It needs some of its functions of it. 


# Train
- recognizeStokes.py - the main file to run the training. 
- CNNarchitectures.py - file for the various CNN architectures. 
- data_preprocessing.py - data pre-processing scripts.
- functions1.py - utility functions for the project. 
- plotFunctions.py - functions for visualizations. 


# Create images using Matlab and discretisation. 
- createManyFracImages.m - key file to create images for the training

# References
@article{kubeyev2023enhancing,
  title={Enhancing multi-physics modelling with deep learning: Predicting permeability through structural discontinuities},
  author={Kubeyev, Amanzhol},
  journal={Engineering Applications of Artificial Intelligence},
  volume={124},
  pages={106562},
  year={2023},
  publisher={Elsevier}
}