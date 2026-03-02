class SGD:
    def __init__(self, net, lr):
        self.net = net
        self.lr = lr

    def step(self):
        for param, grad in self.net.get_parameters():
            if grad is not None:
                param -= self.lr * grad
