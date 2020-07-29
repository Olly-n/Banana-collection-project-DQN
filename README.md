# Project: Navigation and Banana Collection. 
-----





## Description
For this project an agent was trained from scratch using a DQN method to navigate around and collect yellow bananas in an open world environment.
The following is an example of a trained agent in the environment.

![](Gifs/preview2.gif)




## Problem Statement
### Environment and Rewards:
An open space surrounded by four solid walls containing randomly distributed blue and yellow bananas surrounded by four solid walls.
A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana. The goal of your agent is to collect as many yellow bananas as possible while avoiding blue bananas.
### State and Action Spaces:
The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction. Given this information, the agent has to learn how to best select actions. Four discrete actions are available, corresponding to:

| Input  | Action        
| ------ | ------
| `0`    |move forward
| `1`    | move backward
| `2`    | turn left
| `3`    | turn right



### Solution:
The task is episodic with 1000 steps per episode, and is classed as solved when an agent can achieve an average score of +13 over 100 consecutive episodes.

## Dependencies
To run this code you must run an environment with Python 3.6 kernel and the dependencies listed in `requirements.txt`. 

Running the following line will install the required dependencies:
```
pip install -r requirements.txt
``` 

Additionally you will need to download the unity environment. Instruction for this can be found within the `DQN_navigation.ipynb` notbook or at [this link](https://github.com/udacity/deep-reinforcement-learning/tree/master/p1_navigation).

## Files
This repository contains the following files and folders: <br>
`Gifs` : A folder containing gifs of a trained agent. <br>
`Graphs` : A folder containing graphs used to evaluate a trained agent. <br>
`dqn_banana_agent.py` : A file containing <br>
`DQN_navigation.ipynb` : A note book to train, test and evaluate models. <br>
`Trained_agent_best.pth` : A saved model that has learnt to solve the environment <br>

## Running
Please open and follow the instruction in the DQN_navigation.ipynb.

## Results
An agent was trained using the DQN algorithm using Q network with the architecture described in the following table:

| Layer |  Size  
|  ---- | ------
| Input (Observed State) |  37
| Hidden layer 1  | 64
| Hidden layer 2  | 32
| Hidden layer 3  | 16
| Output (Action) | 4

where hidden layer is fully connected and followed by a ReLU activation function.

Using this achitecture the agent was able to solve the problem after 653 episodes. as can be seen in the following graph.
![](Graphs/Solving_criteria.png)

Finally here are a some gifs of the final agent in action.
![](Gifs/preview1.gif)


![](Gifs/preview3.gif)

