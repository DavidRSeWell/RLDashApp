# -*- encoding: utf-8 -*-

import numpy as np
import pandas as pd
import random

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras import backend as K
from collections import deque
from sklearn.metrics.pairwise import cosine_similarity


def get_similar_vectors(state,state_feature_vector ,layer_weights ,num_samples ,top_n):
    '''
        Takes in a state and returns the top x
        most similar states
    '''
    sampled_feature_vectors, sample_states = sample_state_space(state,num_samples,layer_weights)

    cosine_scores = []
    for i ,j in enumerate(sampled_feature_vectors):

        cosine_score = cosine_similarity(state_feature_vector ,j)

        cosine_scores.append([i ,cosine_score])

    cosine_score_df = pd.DataFrame(cosine_scores ,columns=["index" ,"score"])

    cosine_score_df = cosine_score_df.sort_values(by=["score"] ,ascending=False).iloc[:top_n]

    top_indexes = cosine_score_df["index"].tolist()

    top_samples = [sample_states[top_index] for top_index in top_indexes]

    return top_samples

def sample_state_space(state,num_samples, layer_weights):
    '''
        Want to be able to randomly sample from the space
        this will be game dependent. In the case of blackJack
        Just sample a dealer hand and then a random sum for the
        player between 4 and 21 and then sample
    '''
    sample_list = []
    while len(sample_list) < num_samples:

        dealer_hand = np.random.randint(1, 11)
        player_hand = np.random.randint(4, 21)
        ace_or_not = np.random.binomial(1, (1.0 / 13.0))

        sampled_state = [dealer_hand, player_hand, ace_or_not]

        if (sampled_state == state) or (sampled_state in sample_list): # dont sample the same state as your interested in
            continue

        sample_list.append(sampled_state)

    feature_vectors = np.matmul(np.array(sample_list), layer_weights)

    return feature_vectors, sample_list


# Using the player that we use in blackjack
class DeepPlayer:
    # simple DQN agent
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.actions = {0: 'HIT', 1: 'STAY'}
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # discount rate
        self.epsilon = 0.65  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(4 * self.state_size, input_dim=self.state_size, activation='relu',
                        use_bias=False))
        model.add(Dense(4 * self.state_size, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def make_play(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action


if __name__ == '__main__':


    run_test_blackjack_similarity_state = 1
    if run_test_blackjack_similarity_state:

        player = DeepPlayer(3,2)

        keras_model_path = '/Users/befeltingu/DeepRL/Data/models/blackjack_simple_model.h5'

        player.model.load_weights(keras_model_path)

        layer_1_w = K.eval(player.model.layers[0].weights[0])

        state_1 = np.array([[10, 5, 0]])

        state_feature_vector = np.matmul(state_1,layer_1_w)

        state = list(state_1[0])

        top_hands = get_similar_vectors(state,state_feature_vector, layer_1_w, 339, 10)

        print("Current State")
        print(str(state_1))
        print("Most similar hands to the current state")
        print(str(top_hands))