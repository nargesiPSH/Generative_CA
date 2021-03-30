# Generative_CA
This project contains the implementation of using the generative model for continuous authentication on motion sensor data from the H-MOG dataset.

Download H-MOG Dataset from http://www.cs.wm.edu/~qyang/hmog.html and place the zip file as it is into the repository root.

Please make sure to update all the file path in the code to your local directory.

Special thanks to Buech for his work that encouraged me to build my research on top of his research outcome in [1].
Foobox [2] is used To generate adversarial attacks, namely, Deepfool, Boundary and Fast gradient attack. The detail is available in the CA-attack.py file.
To create the generative attacks, we applied TimeGAN [3].

References:

[1] https://github.com/dynobo/ContinAuth

[2] https://github.com/bethgelab/foolbox

[3] https://github.com/jsyoon0823/TimeGAN
