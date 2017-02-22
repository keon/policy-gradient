import gym
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Reshape, Flatten
from keras.optimizers import Adam
from keras.layers.convolutional import Convolution2D


class PGAgent:
    # Inspired by https://gist.github.com/karpathy/a4166c7fe253700972fcbc77e4ea32c5
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = 0.99
        self.learning_rate = 0.001
        self.xs = []
        self.dlogps = []
        self.drs = []
        self.probs = []
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Reshape((1, 80, 80), input_shape=(self.state_size,)))
        model.add(Convolution2D(32, 9, 9, subsample=(4, 4), border_mode='same',
                                activation='relu', init='he_uniform'))
        model.add(Flatten())
        model.add(Dense(128, activation='relu', init='he_uniform'))
        model.add(Dense(64, activation='relu', init='he_uniform'))
        model.add(Dense(self.action_size, activation='softmax'))
        opt = Adam(lr=self.learning_rate)
        model.compile(loss='categorical_crossentropy', optimizer=opt)
        return model

    def remember(self, state, action, prob, reward):
        y = np.zeros([self.action_size])
        y[action] = 1
        self.dlogps.append(np.array(y).astype('float32') - prob)
        self.xs.append(state)
        self.drs.append(reward)

    def act(self, state):
        state = state.reshape([1, state.shape[0]])
        aprob = self.model.predict(state, batch_size=1).flatten()
        self.probs.append(aprob)
        prob = aprob / np.sum(aprob)
        action = np.random.choice(self.action_size, 1, p=prob)[0]
        return action, prob

    def discount_rewards(self, rewards):
        discounted_rewards = np.zeros_like(rewards)
        running_add = 0
        for t in reversed(range(0, rewards.size)):
            if rewards[t] != 0:
                running_add = 0
            running_add = running_add * self.gamma + rewards[t]
            discounted_rewards[t] = running_add
        return discounted_rewards

    def train(self):
        epdlogp = np.vstack(self.dlogps)
        epr = np.vstack(self.drs)
        rewards = self.discount_rewards(epr)
        rewards = rewards / np.std(rewards - np.mean(rewards))
        epdlogp *= rewards
        # Prepare the training batch
        X = np.squeeze(np.vstack([self.xs]))
        y = np.squeeze(np.vstack([epdlogp]))
        Y = self.probs + self.learning_rate * y
        self.model.train_on_batch(X, Y)
        # Clear the batch
        self.xs, self.probs, self.dlogps, self.drs = [], [], [], []

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


def pong_preprocess_screen(I):
    I = I[35:195]
    I = I[::2, ::2, 0]
    I[I == 144] = 0
    I[I == 109] = 0
    I[I != 0] = 1
    return I.astype(np.float).ravel()


if __name__ == "__main__":
    env = gym.make("Pong-v0")
    state = env.reset()
    prev_x = None
    score = 0
    episode = 0

    state_size = 80 * 80
    action_size = env.action_space.n
    agent = PGAgent(state_size, action_size)
    print('loading...')
    agent.load('pong.h5')

    while True:
        env.render()

        # Preprocess, consider the frame difference as features
        cur_x = pong_preprocess_screen(state)
        x = cur_x - prev_x if prev_x is not None else np.zeros(state_size)
        prev_x = cur_x

        # Sample action
        action, prob = agent.act(x)

        state, reward, done, info = env.step(action)
        score += reward

        agent.remember(x, action, prob, reward)

        if done:
            episode += 1
            agent.train()
            print('Episode: %d - Score: %f.' % (episode, score))
            score = 0
            state = env.reset()
            prev_x = None
            if episode > 1 and episode % 10 == 0:
                agent.save('pong.h5')
