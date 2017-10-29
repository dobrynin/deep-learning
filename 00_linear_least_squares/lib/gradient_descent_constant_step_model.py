from .gradient_descent_model import GradientDescentModel
import numpy as np

def norm(v):
    return np.sqrt(v.dot(v))

class GradientDescentConstantStepModel(GradientDescentModel):
    def improve(self, dataset):
        gradient = np.array(self.gradient(dataset))
        gradient = gradient / norm(gradient)

        self.theta0 -= self.learning_rate * gradient[0]
        self.theta1 -= self.learning_rate * gradient[1]

        return (self.theta0, self.theta1)
