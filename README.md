# Emoji Prediction using BiLSTM Model
## Introduction
This project aims to predict appropriate emojis for given text input using a Bidirectional Long Short-Term Memory (BiLSTM) model. Emojis add a layer of emotion and context to text, enhancing the overall communication experience. The BiLSTM model learns from a dataset containing text and corresponding emojis, enabling it to predict suitable emojis for new text inputs.

In this project, we leverage the power of BiLSTM, a type of recurrent neural network, to predict emojis for given text sentences. The BiLSTM architecture allows the model to capture dependencies in both forward and backward directions, making it effective in understanding the context of the text.

## Requirements

To run this project, you need the following dependencies:

- Python
- TensorFlow
- NumPy
- pandas
- Pre-Trained Glove Vector

## Installation

1. Clone this repository to your local machine.
2. Install the required dependencies using pip:

```bash
pip install tensorflow numpy pandas
````
## Model Details
This model is a Bidirectional LSTM (BiLSTM) neural network for emoji prediction. The model has total 7 components. Let's break down each component:

1. `Sequential`: The `Sequential` class from Keras is used to define a linear stack of layers, where each layer follows the previous one.

2. `Bidirectional(LSTM(units=64, return_sequences=True))`: This line adds the first layer, which is a Bidirectional LSTM layer with 64 units. The Bidirectional wrapper makes the LSTM layer process the input data in both forward and backward directions, capturing context from both sides. The `return_sequences=True` argument indicates that the output of this layer will be a sequence rather than a single output for each input.

3. `Dropout(0.5)`: Dropout is a regularization technique used to prevent overfitting in neural networks. It randomly sets a fraction of input units to 0 during training. Here, a dropout rate of 0.5 means that 50% of the units will be dropped out during training.

4. `Bidirectional(LSTM(units=64))`: The second layer is another Bidirectional LSTM layer with 64 units. This layer does not have the `return_sequences=True` argument, which means it will produce a single output for each input, rather than a sequence.

5. `Dropout(0.5)`: Similar to the previous dropout layer, this dropout layer is applied after the second BiLSTM layer.

6. `Dense(units= 10, activation='tanh')`: This is a fully connected dense layer with 10 units and a hyperbolic tangent (tanh) activation function. The `tanh` activation function squashes the output between -1 and 1, making it suitable for handling numerical data within this range.

7. `Dense(units= 5, activation= 'softmax')`: The final dense layer has 5 units and uses the softmax activation function. This layer is responsible for predicting probabilities for each of the 5 possible emojis in the output.

Overall, this BiLSTM model takes a sequence of text as input, processes it through two Bidirectional LSTM layers with dropout to capture context from both directions, and then predicts the probabilities of 5 different emojis using a fully connected layer with a tanh activation function followed by a softmax layer.