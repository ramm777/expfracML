# Enhancing Multi-Physics Modelling with Deep Learning: Predicting Permeability through Structural Discontinuities 
## Here, I present a method of utilizing deep learning to enhance physics-based solid-fluid mechanics modelling by replacing one part of it, permeability through complex geometries. 

Abstract:
Numerical modelling of complex structural discontinuities such as fractures poses a computational challenge, as it involves solving multi-scale and multi-physics phenomena and simulating various processes, including solid, fluid, thermal and chemical interactions. To overcome the limitations of long computation times, simplifications or conceptualizations are often required. However, in multi-physics modelling, it is desirable to obtain predictions of certain parameters without making simplifications. In this study, a data-driven deep learning approach is presented that predicts physical permeability parameters through discontinuities with complicated geometries based on digital images. Images of fractures were generated from a digitalized rough fracture surface of subsurface rock. Permeability was calculated using the Stokes equation and Finite Volume discretization for training and testing purposes. Two cases were analyzed: when the fluid velocity field of the fracture was provided to the CNN for training, and a more challenging case when it was not. Results show that deep learning can accurately predict permeability without fluid velocity information. Besides, the model generalizes well, providing accurate predictions of permeability for fractures with significantly different roughness parameters. In conclusion, this approach can reduce computation time during multi-physics modelling and can be used to predict continuous physical permeability values from an image of a fracture with a complex surface. 

# Requirements

- Python and dependencies --> please see the requirements.txt file for the full requirements of each module. 
- Matlab --> 2017 or above, MRST-2017a or above (Matlab Reservoir Simulation Toolbox https://www.sintef.no/projectweb/mrst/). 


# Installation 

- Matlab --> once installed MRST, please place the Matlab code to the "vem" module inside Mrst. It needs some of its functions of it. 


# Train
- recognizeStokes.py - the main file to run the training. 
- CNNarchitectures.py - file for the various CNN architectures. 
- dataEngineeringStokes.py - some scripts to pre-process data.
- functions1.py - utility functions for the project. 
- plotFunctions.py - functions for visualizations. 


# Create images using Matlab and discretisation. 
- createManyFracImages.m - key file to create images for the training
# References
