import torch
import torch.nn as nn
import torch.nn.functional as F

class ArgmaxApproximator(nn.Module):
    def __init__(self, n, delta, epsilon=1e-2, epsilon1=None):
        """
        n: input dimension
        delta: minimal gap between max and second max for guarantee
        epsilon: desired output error bound
        epsilon1: intermediate approximation error for h_ij (if None, set to epsilon1 = min(0.5, delta/(2*n)))
        """
        super().__init__()
        self.n = n
        self.delta = delta
        # set intermediate epsilon1
        if epsilon1 is None:
            epsilon1 = min(0.5, delta / (2 * n))
        self.epsilon1 = epsilon1
        self.epsilon = epsilon

        # compute alpha to satisfy sigma(alpha * delta) = 1 - epsilon1
        # logistic sigmoid: sigma(t) = 1/(1+exp(-t)); sigma(alpha*delta)=1-epsilon1 => exp(-alpha*delta)=epsilon1/(1-epsilon1)
        self.alpha = (1.0 / delta) * torch.log((1 - epsilon1) / epsilon1)

        # threshold T
        self.T = (n - 1) / 2.0
        # compute beta to satisfy sigma(beta*( (n-1)*(0.5 - epsilon1) )) = 1 - epsilon
        margin = (n - 1) * (0.5 - epsilon1)
        self.beta = (1.0 / margin) * torch.log((1 - epsilon) / epsilon)

        # build first linear layer: weight shape (n*(n-1), n)
        rows = []
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                row = torch.zeros(n)
                row[i] = 1.0
                row[j] = -1.0
                rows.append(row)
        W1 = torch.stack(rows, dim=0) * self.alpha
        self.layer1 = nn.Linear(n, n*(n-1), bias=False)
        self.layer1.weight.data = W1

        # build second linear layer: weight shape (n, n*(n-1)), bias shape (n,)
        W2 = torch.zeros(n, n*(n-1))
        for i in range(n):
            # hidden units h_{i,j} have indices from i*(n-1) to i*(n-1)+(n-1), but careful skip j==i.
            # We can recompute indices: index k corresponds to rows list order above.
            # Alternatively, use block structure:
            for j in range(n):
                if i == j:
                    continue
                # find index in rows list: count previous rows
                idx = i * (n - 1) + (j if j < i else j - 1)
                W2[i, idx] = self.beta
        b2 = - self.beta * self.T
        self.layer2 = nn.Linear(n*(n-1), n)
        self.layer2.weight.data = W2
        self.layer2.bias.data = b2

    def forward(self, x):
        # x: (batch, n)
        h = torch.sigmoid(self.layer1(x))
        y = torch.sigmoid(self.layer2(h))
        return y

# Usage example:
if __name__ == '__main__':
    n = 4
    delta = 0.5
    model = ArgmaxApproximator(n, delta)
    # random input with guaranteed gap
    x = torch.tensor([[1.0, 0.1, -0.2, 0.0], [0.3, 0.8, -0.1, 0.2]])
    y = model(x)
    print(y)
    # expected: approx one-hot vectors
