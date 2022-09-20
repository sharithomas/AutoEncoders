# AutoEncoders
Autoencoder is an unsupervised artificial neural network that is trained to copy its input to output. In the case of image data, the autoencoder will first encode the image into a lower-dimensional representation, then decodes that representation back to the image. Encoder-Decoder automatically consists of the following two structures:
The encoder- This network downsamples the data into lower dimensions.
The decoder- This network reconstructs the original data from the lower dimension representation.

Here, we use this Enoder-Decoder Architecture of Autoencoder for 2 applications - Line Removal from from handwritten digits and Resolution Enhancmnet of digits

for both these applications training with 50k samples of mnist multidigit datas, which are created with program code mnist_multidigit_creation.py
