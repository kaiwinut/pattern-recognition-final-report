import numpy as np

class Classifier:
    def __init__(self):
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train):
        return NotImplementedError()

    def predict(self, X_test):
        return NotImplementedError()


class NCM(Classifier):
    def __init__(self):
        self.weight = None
        self.bias = None

    def fit(self, X_train, y_train):
        p_mean = np.mean(X_train[y_train == 1], axis = 0)
        n_mean = np.mean(X_train[y_train == 0], axis = 0)
        self.weight = 2 * (p_mean - n_mean)
        self.bias = np.sum(n_mean ** 2) - np.sum(p_mean ** 2)

    def predict(self, X_test):
        return (np.dot(X_test, self.weight) + self.bias) >= 0


class LinearSVM(Classifier):
    def __init__(self, lam = 0.1, n_iters = 100, lr = 1e-2):
        self.lam = lam
        self.n_iters = n_iters
        self.lr = lr

    def fit(self, X_train, y_train):
        # Initialize weights
        self.weight = np.zeros(X_train.shape[1] + 1, dtype = np.float32)
        # SGD learning
        for _ in range(self.n_iters):
            for i in range(X_train.shape[0]):
                x = np.concatenate((X_train[i], np.ones(1, dtype = np.float32)))
                y = y_train[i] * 2 - 1

                hinge_loss = max(0, 1 - y * np.dot(self.weight, x))
                self.weight[:-1] = (1 - self.lam * self.lr) * self.weight[:-1]

                if hinge_loss > 0:
                    self.weight += self.lr * x * y

    def predict(self, X_test):
        return np.dot(np.concatenate((X_test, np.ones((X_test.shape[0], 1), dtype = np.float32)), axis = 1), self.weight) >= 0


class KNN(Classifier):
    def __init__(self, k = 1):
        super().__init__()
        self.k = k

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        # Calculate the euclidean distance between each train data and the test data
        dist = np.sum(self.X_train ** 2, axis = 1, keepdims=True) - 2 * np.dot(self.X_train, X_test.T)
        # Let the nearest k train data points vote
        return np.where(np.mean(self.y_train[np.argsort(dist.T, axis=1)[:, :self.k]], axis = 1) >= 0.5, 1, 0)

class MLP:
    def __init__(self):
        self.layers = []
        self.trainable_layers = []
        self.softmax_output = None
        self.train_history = []
        self.val_history = []

    def add(self, layer):
        self.layers.append(layer)

    def set(self, *, loss = None, optimizer = None, accuracy = None):
        if loss is not None:
            self.loss = loss
        if optimizer is not None:
            self.optimizer = optimizer
        if accuracy is not None:
            self.accuracy = accuracy

    def configure(self):
        # Initialize and link layers
        self.input_layer = Input()
        for i in range(len(self.layers)):
            # The first hidden layer
            if i == 0:
                self.layers[i].prev = self.input_layer
                self.layers[i].next = self.layers[i + 1]
            elif i < len(self.layers) - 1:
                self.layers[i].prev = self.layers[i - 1]
                self.layers[i].next = self.layers[i + 1]
            else:
                self.layers[i].prev = self.layers[i - 1]
                self.layers[i].next = self.loss
                self.output_layer = self.layers[i]
            # Initialize trainable layers
            if hasattr(self.layers[i], 'weights'):
                self.trainable_layers.append(self.layers[i])
        # Set trainable layers attribute of loss class
        if self.loss is not None:
            self.loss.set_trainable_layers(self.trainable_layers)
        if isinstance(self.layers[-1], Softmax) and isinstance(self.loss, CategoricalCrossEntropy):
            self.softmax_output = Softmax_CategoricalCrossEntropy()

    def get_params(self):
        params = []
        for layer in self.trainable_layers:
            params.append(layer.get_params())
        return params

    def set_params(self, params):
        for param_set, layer in zip(params, self.trainable_layers):
            layer.set_params(*param_set)

    def forward(self, X, training):
        # Get input
        self.input_layer.forward(X, training)
        # Call forawrd function of every layer. Input is the output of the previous layer
        for layer in self.layers:
            layer.forward(layer.prev.output, training)
        # Return output of the output layer
        return layer.output

    def backpropagate(self, output, y):
        # If backpropagation at the output layer can be performed faster, do it
        if self.softmax_output is not None:
            self.softmax_output.backpropagate(output, y)
            self.layers[-1].dinputs = self.softmax_output.dinputs
            for layer in reversed(self.layers[:-1]):
                layer.backpropagate(layer.next.dinputs)
            return None
        # Normal backpropagation
        self.loss.backpropagate(output, y)
        for layer in reversed(self.layers):
            layer.backpropagate(layer.next.dinputs)

    def train(self, X, y, *, epochs = 1, batch_size = None, summarize_every = 1, verbose = 1, validation_data = None):
        # Calculate train steps from data size and batch size
        steps = 1 if batch_size is None else np.ceil(len(X) // batch_size).astype(int)
        # Training loop
        for epoch in range(1, epochs + 1):
            # Reset loss function / accuracy function parameters
            self.loss.reset_params()
            self.accuracy.reset_params()
            # Train model with assigned batch size
            for step in range(steps):
                # Split data to batches
                X_batch = X if batch_size is None else X[step * batch_size : (step + 1) * batch_size]
                y_batch = y if batch_size is None else y[step * batch_size : (step + 1) * batch_size]
                # Forward the batch to obtain the output
                output = self.forward(X_batch, training = True)
                # Calculate loss
                loss = self.loss.calculate(output, y_batch)
                # Get predictions and calculate accuracy
                preds = self.output_layer.predict(output)
                accuracy = self.accuracy.calculate(preds, y_batch)
                # Backpropagate and update parameters
                self.backpropagate(output, y_batch)
                self.optimizer.prepare_param_update()
                for layer in self.trainable_layers:
                    self.optimizer.update_params(layer)
                self.optimizer.end_param_update()
                # Summarize step
                if not step % summarize_every or step == steps - 1:
                    print(f'Epoch: {epoch} /  Step: {step} / Accuracy: {accuracy:.3f} / Loss: {loss:.3f}')

            # Summarize epoch
            epoch_loss = self.loss.calculate_accumulated()
            epoch_accuracy = self.accuracy.calculate_accumulated()
            if verbose:
                print(f'Training (Accuracy: {epoch_accuracy:.3f} / Loss: {epoch_loss:.3f})')
                self.train_history.append({'epoch': epoch, 'acc': epoch_accuracy, 'loss': epoch_loss})
            if validation_data is not None:
                val_loss, val_acc = self.evaluate(*validation_data, batch_size = batch_size)
                self.val_history.append({'epoch': epoch, 'acc': val_acc, 'loss': val_loss})

    def evaluate(self, X_val, y_val, *, batch_size = None):
        # Calculate validation steps from data size and batch size
        steps = 1 if batch_size is None else np.ceil(len(X_val) // batch_size).astype(int)
        # Reset loss function / accuracy function parameters
        self.loss.reset_params()
        self.accuracy.reset_params()
        # Train model with assigned batch size
        for step in range(steps):
            # Split data to batches
            X_batch = X_val if batch_size is None else X_val[step * batch_size : (step + 1) * batch_size]
            y_batch = y_val if batch_size is None else y_val[step * batch_size : (step + 1) * batch_size]
            # Forward data
            output = self.forward(X_batch, training = False)
            # Calculate loss
            self.loss.calculate(output, y_batch)
            # Get predictions and calculate accuracy
            preds = self.output_layer.predict(output)
            accuracy = self.accuracy.calculate(preds, y_batch)
        # Summarize evaluation
        val_loss = self.loss.calculate_accumulated()
        val_acc = self.accuracy.calculate_accumulated()
        print(f'Validation (Accuracy: {val_acc:.3f} / Loss: {val_loss:.3f})')
        return val_loss, val_acc

    def predict(self, X_test):
        return np.argmax(self.forward(X_test, training = False), axis = 1)

class Layer:
    def get_params(self):
        return self.weights, self.biases

    def set_params(self, weights, biases):
        self.weights = weights
        self.biases = biases

class Input(Layer):
    def forward(self, inputs, training):
        self.output = inputs

class Dense(Layer):
    def __init__(self, n_inputs, n_neurons):
        # Initialize weights and biases
        self.weights = 0.1 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs, training):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

    def backpropagate(self, dvalues):
        # Calculate gradients and update
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis = 0, keepdims = True)
        # Calculate gradients on values
        self.dinputs = np.dot(dvalues, self.weights.T)

class Loss:
    def set_trainable_layers(self, trainable_layers):
        self.trainable_layers = trainable_layers

    def calculate(self, output, y):
        loss = self.forward(output, y)
        self.accumulated_sum += np.sum(loss)
        self.accumulated_count += len(loss)
        return np.mean(loss)

    def calculate_accumulated(self):
        return self.accumulated_sum / self.accumulated_count

    def reset_params(self):
        self.accumulated_sum = 0
        self.accumulated_count = 0

class CategoricalCrossEntropy(Loss):
    def forward(self, preds, y):
        # Prevent division by 0
        preds_clipped = np.clip(preds, 1e-7, 1 - 1e-7)
        # If labels are one hot encoded
        correct_confidences = np.sum(preds_clipped * y, axis = 1) if len(y.shape) == 2 else preds_clipped[range(len(preds)), y]
        # Return loss
        return -np.log(correct_confidences)

    def backpropagate(self, dvalues, y):
        # Change labels to one hot encoded if they are not
        if len(y.shape) == 1:
            y = np.eye(len(dvalues[0]))[y]
        # Calculate gradient
        self.dinputs = -y / dvalues
        # Normalize gradient
        self.dinputs /= len(dvalues)

class Activation:
    def __init__(self):
        pass

class Softmax(Activation):
    def forward(self, inputs, training):
        self.inputs = inputs
        exp_vals = np.exp(inputs - np.max(inputs, axis = 1, keepdims = True))
        probs = exp_vals / np.sum(exp_vals, axis = 1, keepdims = True)
        self.output = probs

    def backpropagate(self, dvalues):
        self.dinputs = np.empty_like(dvalues)
        for i, (single_output, single_dvalue) in enumerate(zip(self.output, dvalues)):
            single_output = single_output.reshape(-1, 1)
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            self.dinputs[i] = np.dot(jacobian_matrix, single_dvalue)

    def predict(self, output):
        return np.argmax(output, axis = 1)

class ReLU(Activation):
    def forward(self, inputs, training):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)

    def backpropagate(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0

    def predict(self, output):
        return output

class Softmax_CategoricalCrossEntropy(Activation):
    def backpropagate(self, dvalues, y_true):
        # If the true y is one hot encoded, return the index of the most likely result
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis = 1)
        # Calculate gradient
        self.dinputs = dvalues.copy()
        self.dinputs[range(len(dvalues)), y_true] -= 1
        # Normalize gradient
        self.dinputs /= len(dvalues)

class Optimizer:
    def __init__(self, learning_rate, decay, eps, beta_1, beta_2):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.eps = eps
        self.beta_1 = beta_1
        self.beta_2 = beta_2
    # Update current learning rate if decay is non zero
    def prepare_param_update(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iterations))
    # Update iteration
    def end_param_update(self):
        self.iterations += 1

class Adam(Optimizer):
    def __init__(self, learning_rate = 0.001, decay = 0., eps = 1e-7, beta_1 = 0.9, beta_2 = 0.999):
        super().__init__(learning_rate, decay, eps, beta_1, beta_2)

    # Ref: https://arxiv.org/pdf/1412.6980v8.pdf
    def update_params(self, layer):
        # Initialize momentums and rms parameters as zeros
        if not hasattr(layer, 'weight_momentums'):
            layer.weight_momentums = np.zeros_like(layer.weights)
            layer.weight_rms = np.zeros_like(layer.weights)
            layer.bias_momentums = np.zeros_like(layer.biases)
            layer.bias_rms = np.zeros_like(layer.biases)
        # Update momentum
        layer.weight_momentums = self.beta_1 * layer.weight_momentums + (1 - self.beta_1) * layer.dweights
        layer.bias_momentums = self.beta_1 * layer.bias_momentums + (1 - self.beta_1) * layer.dbiases
        # Correct momentum to compensate for initial updates since we initialized biases as zeros
        corrected_weight_momentums = layer.weight_momentums / (1 - self.beta_1 ** (self.iterations + 1))
        corrected_bias_momentums = layer.bias_momentums / (1 - self.beta_1 ** (self.iterations + 1))
        # Update rms
        layer.weight_rms = self.beta_2 * layer.weight_rms + (1 - self.beta_2) * layer.dweights ** 2
        layer.bias_rms = self.beta_2 * layer.bias_rms + (1 - self.beta_2) * layer.dbiases ** 2
        # Correct rms to compensate for initial updates since we initialized biases as zeros
        corrected_weight_rms = layer.weight_rms / (1 - self.beta_2 ** (self.iterations + 1))
        corrected_bias_rms = layer.bias_rms / (1 - self.beta_2 ** (self.iterations + 1))
        # Update weights and biases
        layer.weights += -self.current_learning_rate * corrected_weight_momentums / (np.sqrt(corrected_weight_rms) + self.eps)
        layer.biases += -self.current_learning_rate * corrected_bias_momentums / (np.sqrt(corrected_bias_rms) + self.eps)

class Accuracy:
    def __init__(self):
        self.reset_params()

    def reset_params(self):
        self.accumulated_sum = 0
        self.accumulated_count = 0

    def calculate(self, preds, y):
        # Compare predictions and true category
        res = self.compare(preds, y)
        # Update accumulation parameters
        self.accumulated_sum += np.sum(res)
        self.accumulated_count += len(res)
        return np.mean(res)

    def calculate_accumulated(self):
        # Calculate total accuracy
        return self.accumulated_sum / self.accumulated_count

class Categorical(Accuracy):
    def compare(self, preds, y):
        # If y is one hot encoded, return the index of the most likely result
        if len(y.shape) == 2:
            y = np.argmax(y, axis = 1)
        return preds == y