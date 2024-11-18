import nn
from backend import PerceptronDataset, RegressionDataset, DigitClassificationDataset
import numpy as np

class PerceptronModel(object):
    def __init__(self, dimensions: int) -> None:
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.
        """
        self.w = nn.Parameter(1, dimensions)

    def get_weights(self) -> nn.Parameter:
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    def run(self, x: nn.Constant) -> nn.Node:
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)
        """
        "*** TODO: COMPLETE HERE FOR QUESTION 1 ***"
        
        return nn.DotProduct(x, self.w )

    def get_prediction(self, x: nn.Constant) -> int:
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        "*** TODO: COMPLETE HERE FOR QUESTION 1 ***"
        x_scalar = nn.as_scalar(self.run(x))
        if (x_scalar >= 0):
            return 1
        else:
            return -1

    def train(self, dataset: PerceptronDataset) -> None:
        """
        Train the perceptron until convergence.
        """
        "*** TODO: COMPLETE HERE FOR QUESTION 1 ***"
    
        all_correct = False  # Initialisation à False pour commencer l'entraînement

        while not all_correct:  # Boucle jusqu'à convergence
            all_correct = True  # On suppose au départ que tout est correct

            for x, y in dataset.iterate_once(1):  
                y_true = int(nn.as_scalar(y))  
                y_pred = self.get_prediction(x) 

                # Si la prédiction est incorrecte, ajuster les poids
                if y_true != y_pred:
                    
                    # Mise à jour des poids
                    self.w.update(x, nn.as_scalar(y))  
                    all_correct = False  # Au moins une erreur corrigée


class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """

    def __init__(self) -> None:
        # Initialize your model parameters here
        "*** TODO: COMPLETE HERE FOR QUESTION 2 ***"
        
        # Pour la reproductibilité
        np.random.seed(42)

        # Nombre de neurone dans les couches cachées
        self.hidden_layer = 100

        # Poids et biais pour la couche cachée
        self.w_hidden = nn.Parameter(1, self.hidden_layer)
        self.b_hidden = nn.Parameter(1, self.hidden_layer)

        # Poids et biais pour la couche de sortie
        self.w_output = nn.Parameter(self.hidden_layer, 1)
        self.b_output = nn.Parameter(1, 1)


    def run(self, x: nn.Constant) -> nn.Node:
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        "*** TODO: COMPLETE HERE FOR QUESTION 2 ***"
        
        # Calcul de la couche cachée
        hidden = nn.ReLU(nn.AddBias(nn.Linear(x, self.w_hidden), self.b_hidden))
        # Calcul de la sortie
        output = nn.AddBias(nn.Linear(hidden, self.w_output), self.b_output)
        return output

    def get_loss(self, x: nn.Constant, y: nn.Constant) -> nn.Node:
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a loss node
        """
        "*** TODO: COMPLETE HERE FOR QUESTION 2 ***"
        predicted_y = self.run(x)  # Utilise les prédictions
        return nn.SquareLoss(predicted_y, y)

    def train(self, dataset: RegressionDataset) -> None:
        """
        Trains the model.
        """
        "*** TODO: COMPLETE HERE FOR QUESTION 2 ***"
        batch_size = 10
        learning_rate = 0.01
        target_loss = 0.0001  # Seuil de perte pour arrêter l'entraînement
        max_epochs = 10000    # Sécurité pour éviter un entraînement infini

        for epoch in range(max_epochs):  # Limiter le nombre d'epochs
            total_loss = 0
            num_batches = 0

            for x_input, y_golden in dataset.iterate_once(batch_size):
                

                # Calcul de la fonction de perte
                loss = self.get_loss(x_input, y_golden)
                total_loss += nn.as_scalar(loss)  # Accumuler la perte pour calculer la moyenne
                num_batches += 1

                # Mise à jour des poids
                grad_w_hidden, grad_b_hidden, grad_w_output, grad_b_output = nn.gradients(
                    loss, [self.w_hidden, self.b_hidden, self.w_output, self.b_output]
                )

                self.w_hidden.update(grad_w_hidden, -learning_rate)
                self.b_hidden.update(grad_b_hidden, -learning_rate)
                self.w_output.update(grad_w_output, -learning_rate)
                self.b_output.update(grad_b_output, -learning_rate)

            # Calcul de la perte moyenne après une epoch
            avg_loss = total_loss / num_batches
            # print(f"Epoch {epoch + 1}: Average Loss = {avg_loss}")

            # Vérification de la condition d'arrêt
            if avg_loss <= target_loss:
                print("Entraînement terminé : la perte moyenne a atteint le seuil.")
                break
        else:
            print("Entraînement terminé : limite d'epochs atteinte.")                
            


class DigitClassificationModel(object):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """

    def __init__(self) -> None:
        # Initialize your model parameters here
        "*** TODO: COMPLETE HERE FOR QUESTION 3 ***"

    def run(self, x: nn.Constant) -> nn.Node:
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a node with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        "*** TODO: COMPLETE HERE FOR QUESTION 3 ***"

    def get_loss(self, x: nn.Constant, y: nn.Constant) -> nn.Node:
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss node
        """
        "*** TODO: COMPLETE HERE FOR QUESTION 3 ***"

    def train(self, dataset: DigitClassificationDataset) -> None:
        """
        Trains the model.
        """
        "*** TODO: COMPLETE HERE FOR QUESTION 3 ***"
