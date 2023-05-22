# [Robust Non-linear Set-point control with Reinforcement Learning](https://arxiv.org/abs/2304.10277)

Ruoqi Zhang, Per Mattsson, [Torbjörn Wigren](http://www.it.uu.se/katalog/tw) [Uppsala University]

> This paper argues that three ideas can improve reinforcement learning methods even for highly nonlinear set-point control problems: 1) Make use of a prior feedback controller to aid amplitude exploration.  2) Use integrated errors. 3) Train on model ensembles. Together these ideas lead to more efficient training, and a trained set-point controller that is more robust to modelling errors and thus can be directly deployed to real-world nonlinear systems.

This is a PyTorch implementation based on [ElegantRL](https://github.com/AI4Finance-Foundation/ElegantRL) for our ACC paper [Robust Non-linear Set-point control with Reinforcement Learning]()

## Installation and Usage

```
#create virtual env
conda create --name pime python=3.7
source activate pime

#install requirements
pip install -r requirements.txt
```

### How to use our code for new env?

1. Choose your prior controller for exploration, e.g. P-controller or PI-controller
2. Rewrite the env in openAI Gym style 
   1.  contains the integrator in the observation space
   2. contains the model ensemble realted functions. An example is shown below

```python
def reset_changable_parameters(self, params_values):
  self.your_model_ensemble_params = params_values
  
def get_changable_parameters(self):
  return self.your_model_ensemble_params
```

3. Initiaize the final layer of your policy network to zero to start from the prior controller
4. Resample the model param from model ensemble every `n` episodes
