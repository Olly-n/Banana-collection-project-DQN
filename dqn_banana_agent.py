import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from replay_buffers import PrioritisedReplayBuffer, ReplayBuffer


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device = ", device)


class QNetwork(nn.Module):
    """A neural network for Q function approximation."""

    def __init__(self, input_size, output_size, hidden_layers):
        """QNetwork initialiser function.

        Args:
            input_size (int): dimension of state space for input to Q network.
            output_size (int): dimension of action space for value predictions.
            hidden_layers (list[int]): list of dimensions of the hidden layers required.
        """
        super().__init__()
        self.input = nn.Linear(input_size, hidden_layers[0])

        self.hidden_layers = nn.ModuleList()
        for layer_in, layer_out in zip(hidden_layers[:-1], hidden_layers[1:]):
            self.hidden_layers.append(nn.Linear(layer_in, layer_out))
        self.output = nn.Linear(hidden_layers[-1], output_size)

    def forward(self, state):
        """Forward propagation step of Q network.

        Returns a list of estimated q values for each action.
        """

        state = state.float()
        x = self.input(state)
        x = F.relu(x)

        for layer in self.hidden_layers:
            x = layer(x)
            x = F.relu(x)

        actions = self.output(x)

        return actions


class DQNAgent():
    """A deep Q network agent.

    An agent using a deep Q network with a replay buffer, soft target update
    and linear epsilon decay that can learn to solve a task by interacting with
    its environment. Includes option to train using double deep Q learning
    and prioritised replay buffer.
    """

    def __init__(
        self,
        buffer_size,
        seed,
        state_size,
        action_size,
        hidden_layers,
        epsilon,
        epsilon_decay,
        epsilon_min,
        gamma,
        tau,
        learning_rate,
        update_frequency,
        double_Q=False,
        prioritised_replay_buffer=False,
        alpha=None,
        beta=None,
        beta_increment_size=None,
        base_priority=None,
        max_priority=None,
        training_scores=None,
        step_number=0,
    ):
        """DQNAgent initialisation function.

        Args:

            buffer_size (int): maximum size of the replay buffer.
            seed (int): random seed used for batch selection.

            state_size (int): dimension of state space for input to Q network.
            action_size (int): dimension of action space for value predictions.
            hidden_layers (list[int]): list of dimensions for the hidden layers required.

            epsilon (float): probability of choosing non-greedy action in policy.
            epsilon_decay (float): linear decay rate of epsilon with after each step.
            epsilon_min (float): a floor for the decay of epsilon.

            gamma (float): discount factor for future expected returns.
            tau (float): soft update factor used to define how much to shift.
                       target network parameters towards current network parameter.

            learning_rate (float): learning rate for gradient decent optimisation.
            update_frequency (int): how often to update target Q network parameters.

            double_Q (bool): set true to train using double deep Q learning.

            priority_replay_buffer (bool): set true to use priority replay buffer.
            alpha (float): priority scaling hyperparameter.
            beta_zero (float): importance sampling scaling hyperparameter.
            beta_increment_size (float): beta annealing rate.
            base_priority (float): base priority to ensure non-zero sampling probability.
            max_priority (float): initial maximum priority.

            training_scores (list[int]): rewards gained in previous traing episodes. (this is primarily 
                                used to reloading saved agents)
            step_number (int): number of steps the agent has taken. (this is primarily 
                                used to reloading saved agents)

        Notes: Setting tau = 1 will return classic DQN with full target update.
               If using soft updates it is recommended that update frequency is high. 
        """

        self.buffer_size = buffer_size
        self.seed = seed
        if prioritised_replay_buffer:
            self.replay_buffer = PrioritisedReplayBuffer(
                buffer_size,
                alpha,
                beta,
                beta_increment_size,
                base_priority,
                max_priority,
                seed,
            )
        else:
            self.replay_buffer = ReplayBuffer(buffer_size, seed)

        self.state_size = state_size
        self.action_size = action_size
        self.hidden_layers = hidden_layers

        self.Q_net = QNetwork(state_size, action_size, hidden_layers).to(device)
        self.target_Q = QNetwork(state_size, action_size, hidden_layers).to(device)
        self.optimizer = optim.Adam(self.Q_net.parameters(), lr=learning_rate)

        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        self.gamma = gamma
        self.tau = tau

        self.learning_rate = learning_rate
        self.update_frequency = update_frequency

        self.double_Q = double_Q

        self.prioritised_replay_buffer = prioritised_replay_buffer
        self.alpha = alpha
        self.beta = beta
        self.beta_increment_size = beta_increment_size
        self.base_priority = base_priority
        self.max_priority = max_priority

        self.step_number = step_number
        if training_scores is None:
            self.training_scores = []
        else:
            self.training_scores = training_scores

    def step(self, state, action, reward, next_state, done, batch_size):
        """
        A function that records experiences into the replay buffer after each
        environment step, then update the current network parameter and soft
        updates target network parameters.
        """
        self.replay_buffer.add(state, action, reward, next_state, done)
        self.update_Q(batch_size)
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

        self.step_number += 1
        if self.step_number % self.update_frequency == 0:
            self.soft_update_target_Q()

    def act_epsilon_greedy(self, state, greedy=False):
        """ Returns an epsilon greedy action """
        if greedy or random.random() > self.epsilon:
            state = torch.from_numpy(state).unsqueeze(0).to(device)
            self.Q_net.eval()
            with torch.no_grad():
                action_values = self.Q_net.forward(state)
            self.Q_net.train()
            return torch.argmax(action_values).cpu().item()

        return np.random.randint(self.action_size)

    def update_Q(self, batch_size):
        """
        Updates the parameters of the current Q network using backpropagation
        and experiences from the replay buffer.
        """

        if len(self.replay_buffer) > 2*batch_size:

            experience = self.replay_buffer.sample(batch_size)

            states = torch.FloatTensor(experience[0]).to(device)
            actions = torch.LongTensor(experience[1]).unsqueeze(1).to(device)
            rewards = torch.FloatTensor(experience[2]).unsqueeze(1).to(device)
            next_states = torch.FloatTensor(experience[3]).to(device)
            done_tensor = torch.FloatTensor(experience[4]).unsqueeze(1).to(device)

            target_Q_net_max = torch.max(self.target_Q(next_states).detach(), 1, keepdim=True)

            if self.double_Q:
                target_actions = target_Q_net_max[1]
                Q_target_next = self.Q_net(next_states).detach().gather(1, target_actions)
            else:
                Q_target_next = target_Q_net_max[0]

            Q_expected = self.Q_net(states).gather(1, actions)
            Q_target = rewards + self.gamma * Q_target_next * (1 - done_tensor)

            if self.prioritised_replay_buffer:
                idx_list = experience[5]
                weights = torch.FloatTensor(experience[6]).unsqueeze(1).to(device)
                td_error = (Q_target - Q_expected)
                priority_list = torch.abs(td_error.squeeze().detach()).cpu().numpy()
                self.replay_buffer.update(idx_list, priority_list)
                loss = torch.mean((weights*td_error)**2)
            else:
                loss = F.mse_loss(Q_expected, Q_target)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def soft_update_target_Q(self):
        """ Soft updates the target Q network """
        for target_Q_param, Q_param in zip(self.target_Q.parameters(), self.Q_net.parameters()):
            target_Q_param.data = self.tau * Q_param.data + (1 - self.tau) * target_Q_param.data

    def save_agent(self, name, path=""):
        """ Saves agent parameters for loading using load_agent function
        Note: it is torch convention to save models with .pth extension
        """
        params = (
            self.buffer_size,
            self.seed,
            self.state_size,
            self.action_size,
            self.hidden_layers,
            self.epsilon,
            self.epsilon_decay,
            self.epsilon_min,
            self.gamma,
            self.tau,
            self.learning_rate,
            self.update_frequency,
            self.double_Q,
            self.prioritised_replay_buffer,
            self.alpha,
            self.replay_buffer.beta,
            self.beta_increment_size,
            self.base_priority,
            self.max_priority,
            self.training_scores,
            self.step_number,
        )

        checkpoint = {
            "params": params,
            "state_dict": self.Q_net.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }

        path += name
        torch.save(checkpoint, path)


def load_agent(file_path, show_parameter = False):
    """ Reloads agents saved using DQNAgent.save_agent function.

    Args:
        file_path (str): The file path of the saved agent being reloaded.
    """
    checkpoint = torch.load(file_path)
    agent = DQNAgent(*checkpoint["params"])
    agent.Q_net.load_state_dict(checkpoint["state_dict"])
    agent.target_Q.load_state_dict(checkpoint["state_dict"])
    agent.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if show_parameter:
        print("Agent loaded with parameters: \n", agent.__dict__)

    return agent


def criterion_check(agent):
    """Checks if the enviroment has been solved.

    Checks to see if agent recieved a reward of >=13 over 100 consecutive episodes
    and returns the episode number in which this criterion was passed.

   Args:
        agent (DQNAgent): A trained agent.
    """
    train_scores = agent.training_scores
    avg_score = sum(train_scores[:99]) / 100
    i = 100
    for first, last in zip(train_scores[:-99], train_scores[99:]):
        avg_score += last / 100
        if avg_score >= 13:
            print("Criterion passed on episode {}.".format(i))
            return i
        avg_score -= first / 100
        i += 1
    print("Criterion not passed.")
    return None


def train_agent(env, agent, episodes, batch_size, name, path="", print_every=100):
    """ Used to train DQN agents in the a Unity environment.

    Returns list of scores for each episode and additionally saves a copy of
    the final agent and of the agent that scored the highest average score.

    Args:
        env (Unity environment): A Unity environment in which the agent can act.
        agent (DQNAgent): A DQN agent to act in the environment.
        episodes (int): Number of episode to train the agent on
        batch_size (int): Size of experience batch used to train the model
        name (str): Name used to save the best preforming and final agent
        path (str): File path to save the agent to
        print_every (int): How often to print the average score of the agent
    """

    brain_name = env.brain_names[0]

    best_average = -np.Inf
    name_best = name.split(".")[0] + "_best." + name.split(".")[1]

    for i in range(episodes):
        env_info = env.reset(train_mode=True)[brain_name]
        state = env_info.vector_observations[0]
        score = 0
        done = False
        while not done:
            action = agent.act_epsilon_greedy(state)

            env_info = env.step(action)[brain_name]
            next_state = env_info.vector_observations[0]
            reward = env_info.rewards[0]
            done = env_info.local_done[0]
            agent.step(state, action, reward, next_state, done, batch_size)
            score += reward
            state = next_state

        agent.training_scores.append(score)
        if i % print_every == print_every - 1:

            avg_score = np.mean(agent.training_scores[-print_every:])

            if avg_score > best_average:
                best_average = avg_score
                agent.save_agent(name_best, path)
                print("Best agent so far has been saved.")

            print("Episode: {}  Average Score: {}".format(i + 1, avg_score))

    agent.save_agent(name, path)
    print("final agent has been saved.")
    return agent.training_scores


def test_agent(env, agent, episodes, print_scores=False, quick_view=True):
    """Test agent using epsilon greedy policy and returns a list of episode rewards.

    Args:
        env (Unity environment): A Unity environment for the agent to act in
        agent (DQNAgent): A DQN agent to act in the environment
        episodes (int): Number of episodes to test over
        print_scores (bool): Set True to prints episode score after each episode
        quick_view (bool): Set False to run environment in slow mode and observer agents actions
    """
    brain_name = env.brain_names[0]

    episode_scores = []

    for i in range(episodes):
        env_info = env.reset(train_mode=quick_view)[brain_name]
        state = env_info.vector_observations[0]
        score = 0
        done = False
        while not done:
            action = agent.act_epsilon_greedy(state)
            env_info = env.step(action)[brain_name]
            next_state = env_info.vector_observations[0]
            reward = env_info.rewards[0]
            done = env_info.local_done[0]
            score += reward
            state = next_state
        if print_scores:
            print("episode score = ", score)
        episode_scores.append(score)
    print("Average reward over {} tests was {} with a SD of {}".format(
            episodes, np.mean(episode_scores), np.std(episode_scores)
    ))

    return episode_scores
