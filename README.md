# Dense Neural Network *from scratch*

---

## NEURAL NETWORK

### NEURAL NETWORK ARCHITECTURE

- The architechture for the neural network is generalised to perform as a Dense Neural Network.
- The number of Hidden Layers, and the number of Neurons for each hidden layer can be customised as per the requirement.
- The following activation functions - sigmoid, relu, tanh can be set for each Hidden Layer accordingly during initialisation.
- The output layer is considered as a Single Classifier Neuron with sigmoid activation.
- The learning rate can be tweaked according to the requirements and the Adam Optimiser is used to correct the learning rate as epochs increase according to the gradient.
- The following neural network trains te netowrk in batches. Given an input, currently a fraction of the records are sliced for training for each epoch.
- This hence helps in reducing overfitting in most circumstances.

### ADDITIONAL FEATURES

#### Convenient Architechture:

- The following architecture provides an interface to spawn any number of hidden layers with each layer consisting any number of neurons.
- The activation function for each layer can also be customised accordingly.
- Other parameters such as leraning rate, epochs, Adam Optimer parameters can also be tweaked according to the users requirement.

#### Adam Optimiser *(AdaGrad and RMSProp)*:

- We have made use of the adam optimizer algorithm to update the network weights instead of the classical gradient descent.
- It combines the best properties of the AdaGrad and RMSProp optimization algorithms.
- Optimization is achieved using 2 moment parameters first is mean and the second is uncentred variance (m,v).
- It makes use of 2 decay rates beta1 and beta2 to calculate the 2 moment

#### Weight Initialisation Techniques *(He, Xavier)*:

- Weight initialisation plays a crucial role in the training of a neura network as the initialised values get ultimately modified to the value of convergence.
- The following techniques are observed to perform statistically better than random initialisation hence have been employed
- 'He' initialisation has been utilised in order to initialse weights for the hidden layers using 'Sigmoid' and 'Relu' activation.
- 'Xavier' initialisation is applied for the hidden layers which use 'tanh' activation.

#### Batch Processing:

- The training of the neural network is performed batch-wise.
- The batch_ratio can be a fraction (0,1] (default 1) where in for every epoch, the entire data-set can be sliced randomly to that ratio.
- Benefits:- 
  - The neural network is computationally much faster in comparison to training on tuples independently. Bulk intermediate states is stored and matrix multiplication is performed on entire set of tuples.
   - Batch processing also helps in reducing over-fitting considerably.

---

## DATASET

### HYPER-PARAMETERS

- The present neural network configuration is set to consist of 2 Hidden layers
  - 1st Hidden layer = 5, activation='sigmoid'
  - 2nd Hidden layer = 4, activation='sigmoid'
- The current learning rate = 0.01
- Epochs = 350
   - train:test split = 70:30
   - batch-size in fit = (set to default 0.3)
- Adam Optimiser parameters
   - beta1 parameter - 0.9
   - beta2 parameter - 0.99
   - epsilon = 10^-8

### PRE-PROCESSING

<ol>
<li>one hot encoding was used for columns Community, Delivery phase and Residence,
   since they were categorical varibles. 
   NaN values were replaced by mode of the columns
</li>
<li>Since column Education is scaled between 0 - 10 (has predetermined range) so this column is normalized using
   min max normalization.
   NaN values were replaced by mode of the column
</li>
<li>
remaining columns Age,Weight,HB,BP were normalized using standard normalization.
   NaN values were replaced by mean of the columns
</li>
</ol>

---

## CODE EXECUTION

   The original dataset must be present in the folder 'src' with the name ```LBW_Dataset.csv```

   ```bash
   python3 pre_precessing.py
   python3 main.py
   ```

---

## TEAM

- **S Nikhil Ram**
- **Rakshith C**
- **K Vikas Gowda**
  
---