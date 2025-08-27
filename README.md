## Synapse

Synapse is a very minimal DNN library meant for learning.

It implements a PyTorch-style Autograd framework for evaluating gradients during backprop passes.

### Setup


> [!NOTE]
> The repository still in a very alpha stage, please wait until the removal of this message before considering it usable.

To setup the environment (after cloning the repository):

```shell
python3.13 -m venv .synapse && \
source .synapse/bin/activate # change depending on your shell.


pip install -r requirements.txt
pip install -r requirements_test.txt # For testing deps
```


More instructions will be added as we setup the package build and create an installable wheel.
