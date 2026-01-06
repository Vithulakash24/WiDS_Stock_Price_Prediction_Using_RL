## Midterm Assignment
- Solving the Lunar Lander Env using either DQN or DDQN
## Walkthrough
### DQN
- The Feedforward neural net contains 4 layes (1 Input, 2 hidden, 1 output layer)
- The hidden layers contain 128 neurons and output of each layers is passed through ReLu activation fn
### Replay Memory
- Implements experience replay using deque(FIFO)
- Stores the transistion parameters
- The sampling from the buffer is through random sampling
### Hyper parameters
- This class holds all teh hyper parameters required for the model
- Ex: epsilon and epsilon decay, learing rate for the NN, Replay memory size and batch size
### Agent
- The core class that handles the interactions with the environment and trains the policy
- In `__init__` the hyperparameters are loaded and the target and policy network is instantiated
- The `run()` handles the main training loop and it also saves the best model during training and it also returns a reward list
  - The Training loop:
    - Epsilon greedy algorithm is used to pick the action for the agent, initially starts with 100% explorationa and then gradually decreases the prob of random action
    - The action is fed into the env and the transition variables are retured and stored in the replay buffer
    - If there are enough transitions in the replay buffer a batch is sent to the optimize function every step, essentially updating the policy after every step
    - The gradients are then backproagated
- The  `Optimize()` function receives a batch from teh replay buffer and it updates teh policy toward the target policy using the Q-Learning eqn
### Training and Testing of the agent
- The agent is first instantiated then the run function is called with `is_train=True` with teh required number of episodes
- The train run will update the policy and also save the best model
- The agent can be passed with the best model with `is_train=False` for the test run

## Comments
- I tried using DDQN for the agent but the training was very nosiy and unstable so I sticked with DQN
- The hyperparameters used were chosen after testing the aagent with various different parameters
