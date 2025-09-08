📊 Steps Involved
1. Import Libraries

Essential libraries like NumPy, Matplotlib, and TensorFlow are imported.

2. Load and Preprocess Data

Load MNIST dataset from tensorflow.keras.datasets.

Normalize images (scale pixel values between 0 and 1).

Reshape data into (28, 28, 1) for CNN input.

Convert labels into one-hot encoded vectors.

3. Build CNN Model

A sequential CNN model is created with:

Convolutional Layers (filters: 32, 64).

MaxPooling Layers.

Flatten Layer.

Dense Layers with ReLU and Softmax activation.

4. Train the Model

Trained for 5 epochs with validation data.

Optimizer: Adam

Loss Function: Categorical Crossentropy

5. Evaluate the Model

The model is tested on unseen test data and outputs accuracy.

6. Visualize Training

Plots training and validation accuracy per epoch.

7. Prediction on Test Images

A function predict_and_display(image) is used to:

Preprocess an image.

Predict the digit.

Display the image with the predicted label.

🚀 Example Output
Training Accuracy Graph

📈 Training and validation accuracy improves with epochs.

Sample Prediction
sample_index = 10
predicted_label = predict_and_display(test_images[sample_index])
print(f'Actual Label: {np.argmax(test_labels[sample_index])}')


Output:

Predicted Label: 3
Actual Label: 3


The predicted digit matches the actual label.