# Generative_CA
This project contains implementation of using generative model for continues authentication on motion sensor data from H-MOG dateset.

Download H-MOG Dataset from http://www.cs.wm.edu/~qyang/hmog.html and place the zip file as it is into the repository root.

Please make sure to update all the file path in the code to your own local directory.

Special thanks to Buech for his work that encouraged me to build my research on top of his research outcome in [1].
Foobox [2] is used To generate adversarial attacks namely, Deepfool, Boundary and Fast gradient attack. the detail is available in CA-attack.py file.
Ton generate generative attack we applied TimeGAN [3].
References:

[1] https://github.com/dynobo/ContinAuth

[2] https://github.com/bethgelab/foolbox

[3] https://github.com/jsyoon0823/TimeGAN
