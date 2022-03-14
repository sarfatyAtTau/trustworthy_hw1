import numpy as np

class Attack:
    def __init__(self):
        """
        Initializes standard attack parameters. May also include
        the loss, in case a different loss than the model's 
        default should be used by the attack.
        """
        pass

    def execute(self, model, x, y, targeted=False):
        """
        Receives the model, and a batch of samples to produce 
        adversarial examples, and a flag indicating whether the
        attack is targeted or not. In case of untargeted attacks,
        the method returns x_adv with the aim to produce 
        misclassifications into labels other than those specified
        in y. For targeted attacks, y contains the targets, and the
        method aims to return x_adv that are (mis)classified as the
        targets in y. Note that x_adv should be a valid image 
        (i.e., in [0,1]^d).
        """
        pass

class FGSM(Attack):
    """
    The single-step attack of Goodfellow et al. (bounded in
    L_inf-norm).
    """
    def __init__(self, epsilon, loss=None):
        pass

    def execute(self, model, x, y, targeted=False):
        pass

class PGD(Attack):
    """
    The iterative attack of Madry et al. (bounded in L_inf-norm).
    Remember the random initialization in the L_inf-ball
    (there is no need to implement multiple restarts).
    """
    def __init__(self, epsilon, alpha, niters, loss=None):
        pass

    def execute(self, model, x, y, targeted=False):
        pass
