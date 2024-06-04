import torch
import torch.nn as nn
from torchviz import make_dot

# Define a simple model
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_1 = nn.Linear(1, 10)
        self.linear_2 = nn.Linear(10, 1)
        
    def forward(self, x):
        x = self.linear_1(x)
        x = nn.functional.relu(x)
        x = self.linear_2(x)
        x = nn.functional.relu(x)
        return x

# Create an instance of the model
model = MyModel()

# Create some input data
x = torch.tensor([[1.0]], requires_grad=True)

# Compute the output of the model
output = model(x)

make_dot(output).render("output", format="png")

# Compute the gradients of the output with respect to the model parameters
params = list(model.parameters())
grad_params = torch.autograd.grad(output, params, create_graph=True)

grad_params_tensor = torch.cat([grad_param.view(-1) for grad_param in grad_params])
# grad_params_tensor_sum = grad_params_tensor.sum()
grad_params_tensor_sum = torch.norm(grad_params_tensor, p=2)
make_dot(grad_params).render("grad_params", format="png") 
make_dot(grad_params_tensor).render("grad_params_tensor", format="png")
make_dot(grad_params_tensor_sum).render("grad_params_tensor_sum", format="png")


# Compute the gradients of the model's gradients with respect to the input data
grad_grad_input = torch.autograd.grad(grad_params_tensor_sum, x, retain_graph=True)

make_dot(grad_grad_input).render("grad_grad_input", format="png")