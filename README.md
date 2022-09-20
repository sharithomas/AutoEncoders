# AutoEncoders
Autoencoder is an unsupervised artificial neural network. It is trained with orginal image as ouput. Here, first encode the image into a lower-dimensional representation, then decodes that representation back to the orginal image. Encoder-Decoder consists of the following two structures:
The encoder- This network downsamples the data into lower dimensions.
The decoder- This network reconstructs the original data from the lower dimension representation.

Here, use this Enoder-Decoder Architecture of Autoencoder for  Line Removal from from handwritten multidit digits 

For training 50k samples of mnist multidigit datas are using , which are created with program code mnist_multidigit_creation.py

All images are resized to a standard size of  28x256  without distortion and used image padding

#### Model Architecture
Encoder- 3 convolution layer followed with ReLu activation function
Decoder- 3ConvolutionalTranspose followed with ReLu activation function and Sigmoid function 

![image](https://user-images.githubusercontent.com/61357572/191325046-30a3b99a-7e30-4461-9d55-44778e303fb8.png)


For training give multidigit with line(Ocllusion) as input and Multidit without Line(NoOcllusion) as output.

Below Shows some sample output:

![image](https://user-images.githubusercontent.com/61357572/191212878-655c2967-88e7-4097-92a2-b9308bd89311.png)



 
