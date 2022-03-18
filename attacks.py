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
        self.epsilon = epsilon
        self.loss = loss

    def execute(self, model, x, y, targeted=False):
        models_grad = model.compute_dldi(x, y, clear_state=True, loss=self.loss).reshape(x.shape[0], -1)
        if targeted:
            delta = - self.epsilon * np.sign(models_grad)
        else:
            delta = self.epsilon * np.sign(models_grad)

        assert (np.abs(delta) <= self.epsilon).all(), "Adverserial perturbation exceesds it's inf-norm"

        # Clipping x_avd to the [0,1] range
        x_adv = x + delta
        x_adv = np.minimum(x_adv, 1)
        x_adv = np.maximum(x_adv, 0)
        assert (0 <= x_adv).all() and (x_adv <= 1).all(), "Adverserial example created is not in valid image range."
        return x_adv

class PGD(Attack):
    """
    The iterative attack of Madry et al. (bounded in L_inf-norm).
    Remember the random initialization in the L_inf-ball
    (there is no need to implement multiple restarts).
    """
    def __init__(self, epsilon, alpha, niters, loss=None):
        self.epsilon = epsilon
        self.alpha = alpha
        self.niters = niters
        self.loss = loss

    def execute(self, model, x, y, targeted=False):
        initial_random_perturbation = self.epsilon * np.random.randn(x.shape[0], x.shape[1])
        assert initial_random_perturbation.shape == x.shape, "Dimension problem: x.shape !== initial_random_perturbation.shape"
        x_adv = x + initial_random_perturbation

        for _ in range(self.niters):
            models_grad = model.compute_dldi(x_adv, y, clear_state=True, loss=self.loss).reshape(x_adv.shape[0], -1)
            if targeted:
                delta = - self.alpha * np.sign(models_grad)
            else:
                delta = self.alpha * np.sign(models_grad)

            x_adv = x_adv + delta

            # Projecting delta from original x to ensure L_inf-norm restriction
            # projected_delta_from_x = self.epsilon * np.tanh(x_adv - x)
            projected_delta_from_x = np.minimum(self.epsilon, np.maximum(-self.epsilon, x_adv - x))
            x_adv = x + projected_delta_from_x
            assert (np.abs(projected_delta_from_x) <= self.epsilon).all(), "Adverserial perturbation exceesds it's inf-norm"

        # Clipping x_avd to the [0,1] range
        x_adv = np.minimum(x_adv, 1)
        x_adv = np.maximum(x_adv, 0)
        assert (0 <= x_adv).all() and (x_adv <= 1).all(), "Adverserial example created is not in valid image range."

        return x_adv
