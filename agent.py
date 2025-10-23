import torch
import torch.nn as nn
import torch.nn.functional as F
from networks import Encoder, EmbeddingNet
import numpy as np

class Actor1(nn.Module):
    def __init__(self, n_state, n_action, hidden_dim):
        super(Actor1, self).__init__()
        self.linear1 = nn.Linear(n_state, hidden_dim)
        self.mu = nn.Linear(hidden_dim, n_action)
        self.relu = nn.ReLU()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def forward(self, states, fixed_action = None):
        x = self.relu(self.linear1(states))
        mu = (torch.tanh(self.mu(x)) + 1.0) / 2.
        sigma = torch.full((states.shape[0],), 0.1).unsqueeze(1).to(self.device)
        distribution = torch.distributions.Normal(mu, sigma)

        if fixed_action is None:
            action = distribution.sample()
        else:
            action = fixed_action

        log_prob = distribution.log_prob(action)
        action = torch.clamp(action, min=0, max=1)

        return action, log_prob

class Actor2(nn.Module):
    def __init__(self, n_state, n_action, hidden_dim):
        super(Actor2, self).__init__()
        self.linear1 = nn.Linear(n_state, hidden_dim)
        self.mu = nn.Linear(hidden_dim, n_action)
        self.relu = nn.ReLU()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def forward(self, states, fixed_action = None):
        x = self.relu(self.linear1(states))
        mu = torch.tanh(self.mu(x)) / 2.
        sigma = torch.full((states.shape[0],), 0.1).unsqueeze(1).to(self.device)
        distribution = torch.distributions.Normal(mu, sigma)
        actions_value = torch.tensor([0.375, 0.125, -0.125, -0.375]).to(self.device)
        if fixed_action is None:
            action_pdf = torch.exp(distribution.log_prob(actions_value))
            action_probs = F.softmax(action_pdf, dim=-1)
            action = torch.multinomial(action_probs, 1)
            log_prob = torch.log(torch.gather(input=action_probs, dim=1, index=action))

        else:
            action = fixed_action
            action_pdf = torch.exp(distribution.log_prob(actions_value))
            action_probs = F.softmax(action_pdf, dim=-1)
            log_prob = torch.log(torch.gather(input=action_probs, dim=1, index=action))

        return action, log_prob

class Actor3(nn.Module):
    def __init__(self, n_state, n_action, hidden_dim):
        super(Actor3, self).__init__()
        self.linear1 = nn.Linear(n_state, hidden_dim)
        self.mu = nn.Linear(hidden_dim, n_action)
        self.relu = nn.ReLU()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def forward(self, states, fixed_action = None):
        x = self.relu(self.linear1(states))
        mu = (torch.tanh(self.mu(x)) + 1.0) / 2.
        sigma = torch.full((states.shape[0],), 0.1).unsqueeze(1).to(self.device)
        distribution = torch.distributions.Normal(mu, sigma)

        if fixed_action is None:
            action = distribution.sample()
        else:
            action = fixed_action

        log_prob = distribution.log_prob(action)
        action = torch.clamp(action, min=0, max=1)

        return action, log_prob

class Actor4(nn.Module):
    def __init__(self, n_state, n_action, hidden_dim):
        super(Actor4, self).__init__()
        self.linear1 = nn.Linear(n_state, hidden_dim)
        self.mu = nn.Linear(hidden_dim, n_action)
        self.relu = nn.ReLU()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def forward(self, states, fixed_action = None):
        x = self.relu(self.linear1(states))
        mu = (torch.tanh(self.mu(x)) + 1.0) / 2.
        sigma = torch.full((states.shape[0],), 0.1).unsqueeze(1).to(self.device)
        distribution = torch.distributions.Normal(mu, sigma)

        if fixed_action is None:
            action = distribution.sample()
        else:
            action = fixed_action

        log_prob = distribution.log_prob(action)
        action = torch.clamp(action, min=0, max=1)

        return action, log_prob

class Critic(nn.Module):
    def __init__(self, n_state, hidden_dim):
        super(Critic, self).__init__()
        self.linear1 = nn.Linear(n_state, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.relu(self.linear1(x))
        out = self.linear2(out)
        return out


class Meta_MTO_agent(nn.Module):
    def __init__(self,node_dim, embed_dim, hidden_dim):
        super(Meta_MTO_agent, self).__init__()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.embedder = EmbeddingNet(node_dim, embed_dim).to(self.device)
        self.encoder = Encoder(embed_dim, 2*hidden_dim).to(self.device)
        self.actor1 = Actor1(2*embed_dim, 1, hidden_dim).to(self.device)
        self.actor2 = Actor2(2*embed_dim, 1, hidden_dim).to(self.device)
        self.actor3 = Actor3(2*embed_dim, 1, hidden_dim).to(self.device)
        self.actor4 = Actor4(2*embed_dim, 1, hidden_dim).to(self.device)
        self.critic = Critic(2*embed_dim, hidden_dim).to(self.device)
        self.optim = torch.optim.Adam(self.parameters(), lr=3e-4)
        self.eps_clip = 0.2
        self.K_epoches = 3
        self.gamma = 0.99
        self.n_steps = 10

    def get_action(self, x):
        compatibility, out = self.encoder(self.embedder(x.to(self.device)))
        compatibility = compatibility.squeeze(0)
        out = out.squeeze(0)
        select_index = torch.argmax(compatibility, dim=-1)

        states = torch.concatenate((out[0], out[select_index[0]])).unsqueeze(0)
        for i in range(1, select_index.shape[0]):
            temp = torch.concatenate((out[i], out[select_index[i]])).unsqueeze(0)
            states = torch.concatenate((states, temp), dim=0)

        out = states

        actions1, logprobs1 = self.actor1(out)
        actions2, logprobs2 = self.actor2(out)
        actions3, logprobs3 = self.actor3(out)
        actions4, logprobs4 = self.actor4(out)
        output_logprobs = torch.sum(logprobs1 + logprobs2 + logprobs3 + logprobs4)

        output_action = []
        actions = (actions1, actions2, actions3, actions4)
        for i in range(select_index.shape[0]):
            tup = (select_index[i].item(), actions[0][i], actions[1][i], actions[2][i], actions[3][i])
            output_action.append(tup)

        return output_action, actions, output_logprobs

    def get_value(self, x):
        compatibility, out = self.encoder(self.embedder(x.to(self.device)))
        compatibility = compatibility.squeeze(0)
        out = out.squeeze(0)
        select_index = torch.argmax(compatibility, dim=-1)

        states = torch.concatenate((out[0], out[select_index[0]])).unsqueeze(0)
        for i in range(1, select_index.shape[0]):
            temp = torch.concatenate((out[i], out[select_index[i]])).unsqueeze(0)
            states = torch.concatenate((states, temp), dim=0)

        out = states

        states_value = torch.mean(self.critic(out))

        return states_value

    def get_value_detach(self, x):
        with torch.no_grad():
            compatibility, out = self.encoder(self.embedder(x.to(self.device)))
            compatibility = compatibility.squeeze(0)
            out = out.squeeze(0)
            select_index = torch.argmax(compatibility, dim=-1)

            states = torch.concatenate((out[0], out[select_index[0]])).unsqueeze(0)
            for i in range(1, select_index.shape[0]):
                temp = torch.concatenate((out[i], out[select_index[i]])).unsqueeze(0)
                states = torch.concatenate((states, temp), dim=0)

            out = states

            states_value_detach = torch.mean(self.critic(out))

        return states_value_detach

    def get_log_probs(self, x, actions):
        actions = actions
        compatibility, out = self.encoder(self.embedder(x.to(self.device)))
        compatibility = compatibility.squeeze(0)
        out = out.squeeze(0)
        select_index = torch.argmax(compatibility, dim=-1)

        states = torch.concatenate((out[0], out[select_index[0]])).unsqueeze(0)
        for i in range(1, select_index.shape[0]):
            temp = torch.concatenate((out[i], out[select_index[i]])).unsqueeze(0)
            states = torch.concatenate((states, temp), dim=0)

        out = states

        actions1, logprobs1 = self.actor1(out, fixed_action=actions[0])
        actions2, logprobs2 = self.actor2(out, fixed_action=actions[1])
        actions3, logprobs3 = self.actor3(out, fixed_action=actions[2])
        actions4, logprobs4 = self.actor4(out, fixed_action=actions[3])
        output_logprobs = torch.sum(logprobs1 + logprobs2 + logprobs3 + logprobs4)

        return output_logprobs

    def update(self, transition_dict):
        states = torch.tensor(np.array(transition_dict['states']), dtype=torch.float32).to(self.device)
        next_states = torch.tensor(np.array(transition_dict['next_states']),dtype=torch.float32).to(self.device)
        old_logprobs = torch.tensor(transition_dict['logprobs'],dtype=torch.float32).detach().to(self.device)
        actions = transition_dict['action']
        reward = transition_dict['rewards']

        old_value = None
        for k in range(self.K_epoches):
            log_probs = []
            state_values = []
            state_values_detach = []
            for i in range(self.n_steps):
                log_prob = self.get_log_probs(states[i].unsqueeze(0), actions[i])
                state_value = self.get_value(states[i].unsqueeze(0))
                state_value_detach = self.get_value_detach(states[i].unsqueeze(0))
                log_probs.append(log_prob)
                state_values.append(state_value)
                state_values_detach.append(state_value_detach)

            log_probs = torch.stack(log_probs)
            state_values = torch.stack(state_values)
            state_values_detach = torch.stack(state_values_detach)

            Reward = []
            R = self.get_value_detach(next_states[-1].unsqueeze(0))

            for r in reward[::-1]:
                R = R * self.gamma + r
                Reward.append(R)

            Reward = torch.stack(Reward[::-1])

            ratios = torch.exp(log_probs - old_logprobs)

            advantages = Reward - state_values_detach

            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            reinforce_loss = torch.mean(-torch.min(surr1, surr2))

            if old_value is None:
                baseline_loss = ((state_values - Reward) ** 2).mean()
                old_value = state_values.detach()
            else:
                vpredclipped = old_value + torch.clamp(state_values - old_value, - self.eps_clip, self.eps_clip)
                v_max = torch.max(((state_values - Reward) ** 2), ((vpredclipped - Reward) ** 2))
                baseline_loss = v_max.mean()

            total_loss = baseline_loss + reinforce_loss

            self.optim.zero_grad()
            total_loss.backward()

            self.optim.step()
