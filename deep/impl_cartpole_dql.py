import gym
import torch.nn as nn
import torch.nn.functional as F
import random
import collections
import torch
import torch.optim as optim

gamma = 0.98
batch_size = 32
buffer_limit = 50000
learning_rate = 0.0005

class Qnet(nn.Module):
    def __init__(self):
        super(Qnet,self).__init__()
        self.fc1 = nn.Linear(4,128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

    def sample_action(self, obs, epsilon):
        out = self.forward(obs)
        coin = random.random()
        if coin < epsilon:
            return random.randint(0,1)
        else :
            return out.argmax().item()

class ReplayBuffer():
    def __init__(self):
        self.buffer = collections.deque(maxlen=buffer_limit)

    def put(self, transition):
        self.buffer.append(transition)

    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_list, a_list,r_list,s_prime_list, done_mask_list = [],[],[],[],[]

        for transition in mini_batch:
            s, a, r, s_prime, done_mask = transition
            s_list.append(s)
            a_list.append([a])
            r_list.append([r])
            s_prime_list.append(s_prime)
            done_mask_list.append([done_mask])

        return torch.tensor(s_list, dtype=torch.float), torch.tensor(a_list), torch.tensor(r_list), \
               torch.tensor(s_prime_list, dtype=torch.float), torch.tensor(done_mask_list)

    def size(self):
        return len(self.buffer)


def train(q, q_target, memory, optimizer):

    for i in range(10):
        s, a, r, s_prime, done_mask = memory.sample(batch_size)

        q_out = q(s)
        q_a = q_out.gather(1,a)

        max_q_prime = q_target(s_prime).max(1)[0].unsqueeze(1)

        target = r + gamma * max_q_prime * done_mask
        loss = F.mse_loss(q_a,target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def main():
    env = gym.make('CartPole-v1')
    q = Qnet()
    q_target = Qnet()
    q_target .load_state_dict(q.state_dict())
    memory = ReplayBuffer()
    optimizer = optim.Adam(q.parameters(), lr=learning_rate)

    print_interval = 20
    score = 0.0

    for n_epi in range(3000):
        epsilon = max(0.01, 0.08-0.01*(n_epi/200))

        s = env.reset()
        done = False
        while not done:
            a = q.sample_action(torch.from_numpy(s).float(), epsilon)
            s_prime, r, done, info = env.step(a)
            done_mask = 0.0 if done else 1.0

            memory.put((s,a,r/100.0,s_prime,done_mask))
            s = s_prime

            score += r
            if done:
                break

        if memory.size() > 2000:
            train(q, q_target, memory, optimizer)

        if n_epi % print_interval == 0 and n_epi != 0:
            q_target.load_state_dict(q.state_dict())
            print("n_episode : {}, score : {:.1f}, n_buffer: {}, eps: {:.1f}%"
                  .format(n_epi, score/print_interval, memory.size(), epsilon*100))
            score = 0


if __name__ == '__main__':
    main()