# Transfer Learning and Fine-Tuning for Movie Poster Genre Classifications

CMSC421 Introduction to Artificial Intelligence Spring 2024

## Group Members:
Cara Murphy, Jihyo Park, Jason Zhong, Michael Tran, Nitin Kanchinadam

## Motivations and Goals:
For our final project, we performed transfer learning and fine-tuned three existing image classifiers, MobileNetV2, ResNet, and GoogLeNet, to classify movies by their genres according to their posters from our specified IMDb dataset. Image classification in general is a difficult problem to solve because of the wide variety of input data and the resulting need for a large amount of training data. Additionally, we wanted to see if there is a strong correlation between a movie’s poster and its genre. 

Despite the challenges, tackling this issue of image classification is a worthwhile struggle due to its potential in other applications such as object recognition, which can be used for law enforcement to recognize license plates, criminals, and weapons, or in medicine to identify tumors and fractures.

Our initial goal was to fine-tune MobileNetV2 to be able to classify a movie poster into 28 different genres using Keras. We wanted to test various hyperparameters and loss functions for training in such a process. As the project went on, we ended up switching to ResNet and GoogLeNet for increased model complexity. After realizing that classifying 28 genres was too complicated, we adjusted our goals to focus on classifying 6 different genres, choosing the classes with the most images in the dataset.

## Solution
Our solution to this problem was to repurpose existing models, MobileNetV2, ResNet, and GoogLeNet, and optimize them to correctly classify movie genres from their posters.

The first thing we did when developing this model was preprocessing the data. This included dropping rows with missing values or duplicate rows. Each image was retrieved using a GET request with the image URLs and one-hot encoded genre vectors were made to account for the multi-labeled nature of the dataset.

All images were resized to (224, 224) since this size is optimal for the chosen pre-trained models. The aspect ratio was kept the same and the resized images were padded on their smaller dimension to ensure uniform input size. Images were also normalized such that all pixel values are in the range [0, 1) by dividing by 255.0. Data was split into a training, validation, and testing set, which were 60%, 20%, and 20% of the data, respectively.

We decided to reduce the number of classes we were attempting to classify to the 6 classes with the largest number of images in the dataset: Drama, Documentary, Comedy, Action, Thriller, and Horror. This was in part to reduce complexity but also because if we had considered too many classes, we would have been left with a large class imbalance with the least popular classes having far too few posters. We ended up using 12726 distinct images, detailed as follows: {‘Thriller’: 1697, ‘Horror’: 1842, ‘Action’: 2042, ‘Drama’: 2156, 'Comedy': 2176, ‘Documentary’: 2813’}.

While working on this problem, we built upon three pre-trained models: MobileNet, ResNet, and GoogLeNet. MobileNetV2 is a convolutional neural network (CNN) designed first for mobile devices, so it is optimized to use few resources but still yields high accuracy in classification. This increased efficiency was useful in terms of training and testing time despite the large volume of data. However, we quickly realized that MobileNetV2 was not the best model for this problem, and we would need to sacrifice faster training times to ensure higher accuracies and F1 scores.

Next, we decided to modify the ResNet and GoogLeNet models. According to Sharma et al. (2018), those two models are state-of-the-art and performed the best when compared to the AlexNet model, making them the best models for image classification to date.

MobileNet has a 3D output, but since we were doing classification, we needed to convert that output to a 1D result. Consequently, we added a Flatten layer and several Linear layers to the end of the MobileNet model. Adding the last layer ensured that the number of outputs was equal to our desired number of classes. For ResNet and GoogLeNet, only a single Linear layer was added as the outputs of these two models are already 1D.

Our solution utilized Binary Cross Entropy (BCE) Loss for our loss function and Stochastic Gradient Descent for our optimizer. BCE loss (with Sigmoid activation) was chosen because of the multi-label nature of movie posters, allowing us to assign multiple genres to a single image. The Sigmoid function would treat each class as a binary classification and give us a binary probability of an image belonging to that class. The problem with this approach is that the model easily fell into predicting 0’s for all classes. Thus we moved on to using Cross Entropy Loss and Softmax with a threshold to ensure that the model outputs a classification. This led to a worse result and we realized that with multi-label classification, the use of Softmax is not feasible. For the optimization function, gradient descent is an effective optimizer for classification tasks. SGD introduces randomness to gradient descent and is also similar to the Adam optimizer, which is also very good for classification. We tried both optimizers and found minimal differences.


## Results and Evaluation
We tried various combinations of hyperparameters and structures to create a model to fit onto the dataset. Our best model achieved a validation F1 score of 0.403 and a training F1 score of 0.398 using Binary Cross Entropy (BCE) Loss for our loss function and Stochastic Gradient Descent for our optimizer.

We added two linear layers before the final output and applied a sigmoid function on the final output as mentioned. The model actually attempted to predict whether or not a movie poster was of the Thriller genre. For all the other classes, however, it would mostly predict either all 0’s or all 1’s. This indicates that the models we had been training weren’t exactly fitting on the dataset and properly classifying the movie posters.

After coming to this conclusion, we looked back on scholarly articles to see how other researchers classified movie posters. Kundalia et al. used InceptionV3 as their base model and treated the movie classification as a single classification problem. That is, each movie can only belong to one genre. After that, they tested the model on a multi-label dataset by choosing the genres with the highest probabilities for a given movie poster. We attempted to replicate their approach by modifying the dataset and using a Cross Entropy Loss function, but none of our models could accurately classify the movies into a single genre. As a result, we stuck with the previous model.
