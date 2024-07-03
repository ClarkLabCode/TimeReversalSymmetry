# TimeSymmetriesDev
Time reversal symmetries stuff

Contains code for generating moving edge stimuli with various interesting combinations of symmetries in x, c, and t. 
These stimuli will be run on the motion detection rigs as well as put through the neural network in MotionDetectionNonlinearity.

moving_edges.py contains lots of functions for creating and processing moving edge stimuli.

build_stimuli.py uses functions in moving_edges.py to create and show stimuli. Not all commented text has the imports working right now.

Can play around with the model + the designed stimuli in models/results_analysis.ipynb.
