import torch 

class MLP(torch.nn.Module) : 
    # ->none 和 super 起什么作用？
    # none表示输出为空，super是跟继承有关的
    def __init__(self, obs_size, n_act) -> None:
        super().__init__()
        self.mlp = self.__mlp(obs_size, n_act)

    def __mlp(self, obs_size, n_act) :
        return torch.nn.Sequential(
            torch.nn.Linear(obs_size, 50),
            torch.nn.ReLU(),
            torch.nn.Linear(50, 50),
            torch.nn.ReLU(),
            torch.nn.Linear(50, 50),
            torch.nn.ReLU(),
            torch.nn.Linear(50, n_act)
        )

    def forward(self, x) :
        return self.mlp(x)