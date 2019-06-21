# -*- encoding: utf-8 -*-

import random
import gym
import numpy as np
import pandas as pd
import warnings

from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from sklearn.metrics.pairwise import cosine_similarity
from keras import backend as K

warnings.filterwarnings("ignore",category=DeprecationWarning)



def get_similar_vectors(state,state_feature_vector ,layer_weights ,num_samples ,top_n):
    '''
        Takes in a state and returns the top x
        most similar states
    '''

    state = list(state[0])
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


class PolicyIterPlayer:

    def __init__(self,ante_size,bet_size,eval_iters,dealer):

        self.actions = {0:"Call",1:"Fold"}
        self.ante_size = ante_size
        self.bet_size = bet_size
        self.dealer = dealer
        self.evaluation_iterations = eval_iters
        # Mapping of states to action
        self.policy = {}
        self.policy_value = {}

    def init_policy(self):

        '''
        The current state of the game is just the dealers up card
        :return:
        '''

        # init policy
        for i in range(1,11):
            # Ace = 1
            random_action = np.random.randint(0,2)

            self.policy[i] = random_action # we will just be playing pure strategies

            self.policy_value[i] = 0

    def make_play(self,cards):

        sum_cards = np.array(cards).sum()

        if sum_cards == 2:
            return "HIT"

        if 1 in cards:
            sum_cards_use_ace = sum_cards + 10
            if sum_cards_use_ace <= 21:
                sum_cards = sum_cards_use_ace

        if sum_cards < 17:
            return "HIT"

        elif sum_cards > 21:

            return "BUST"

        else:

            return "STAY"

    def evaluate_hand(self,cards):
        sum_cards = np.array(cards).sum()

        if 1 in cards:
            sum_cards_use_ace = sum_cards + 10
            if sum_cards_use_ace <= 21:
                sum_cards = sum_cards_use_ace

        return sum_cards

    def evaluate_policy_0(self):
        '''
        using the algorithm from barton
        2. Policy Evaluation
           Repeat
           ∆ ← 0
           For each s ∈ S:
                v ← V (s)
                V (s) ← P s 0 ,r p(s 0 , r|s, π(s)) r + γV (s 0 ) 
                ∆ ← max(∆, |v − V (s)|)
           until ∆ < θ (a small positive number)

        because in this form of blackjack there is no value of the future state
        it is easier to just simulate the current state a large number of times and
        obtain the average and use the for the value at the state. If the policy at
        the state is to fold then the value is 0
        :return:
        '''

        for state in range(1,11):
            # just evaluate the state assuming you have called
            state_value = 0
            dealer_up_card = state
            for iter in range(self.evaluation_iterations):

                pc1 = self.dealer.deal_and_replace()
                pc2 = self.dealer.deal_and_replace()
                player_hand = [pc1,pc2]

                dealer_down_card = self.dealer.deal_and_replace()

                # now do the same loop for the dealer
                # the dealer is using a different policy
                dealer_cards = [dealer_up_card,dealer_down_card]
                dealer_state = "HIT"
                while (dealer_state == "HIT"):

                    dealer_state = self.dealer.make_play(dealer_cards)

                    if dealer_state == "STAY":
                        break

                    elif dealer_state == "BUST":
                        break

                    else:

                        new_card = self.dealer.deal_and_replace()
                        dealer_cards += [new_card]

                if dealer_state == "BUST":
                    state_value += self.bet_size
                    continue

                dealer_score = self.dealer.current_score

                player_score = self.evaluate_hand(player_hand)

                if player_score > dealer_score:

                    state_value += self.bet_size

                elif dealer_score > player_score:

                    state_value -= self.bet_size

                else:
                    state_value += 0 # its a chop do nothing


            # finished evaluating the current state
            self.policy_value[state] = state_value / float(self.evaluation_iterations) # take the average

    def evaluate_policy_1(self):
        '''
        using the algorithm from barton
        2. Policy Evaluation
           Repeat
           ∆ ← 0
           For each s ∈ S:
                v ← V (s)
                V (s) ← P s 0 ,r p(s 0 , r|s, π(s)) r + γV (s 0 ) 
                ∆ ← max(∆, |v − V (s)|)
           until ∆ < θ (a small positive number)

        because in this form of blackjack there is no value of the future state
        it is easier to just simulate the current state a large number of times and
        obtain the average and use the for the value at the state. If the policy at
        the state is to fold then the value is 0
        :return:
        '''

        for state in range(1,11):
            # just evaluate the state assuming you have called
            state_value = 0
            dealer_up_card = state
            for iter in range(self.evaluation_iterations):

                pc1 = self.dealer.deal_and_replace()
                pc2 = self.dealer.deal_and_replace()
                player_hand = [pc1,pc2]

                dealer_down_card = self.dealer.deal_and_replace()

                # Player just mimics the dealers strategy in this version
                player_state = "HIT"
                while (player_state == "HIT"):

                    player_state = self.make_play(player_hand)

                    if player_state == "STAY":
                        break

                    elif player_state == "BUST":
                        break

                    else:

                        new_card = self.dealer.deal_and_replace()
                        player_hand += [new_card]

                if player_state == "BUST":
                    state_value -= self.bet_size
                    continue

                # now do the same loop for the dealer
                # the dealer is using a different policy
                dealer_cards = [dealer_up_card,dealer_down_card]
                dealer_state = "HIT"
                while (dealer_state == "HIT"):

                    dealer_state = self.dealer.make_play(dealer_cards)

                    if dealer_state == "STAY":
                        break

                    elif dealer_state == "BUST":
                        break

                    else:

                        new_card = self.dealer.deal_and_replace()
                        dealer_cards += [new_card]

                if dealer_state == "BUST":
                    state_value += self.bet_size
                    continue


                dealer_score = self.evaluate_hand(dealer_cards)

                player_score = self.evaluate_hand(player_hand)

                if player_score > dealer_score:

                    state_value += self.bet_size

                elif dealer_score > player_score:

                    state_value -= self.bet_size

                else:
                    state_value += 0 # its a chop do nothing


            # finished evaluating the current state
            self.policy_value[state] = state_value / float(self.evaluation_iterations) # take the average

    def improve_policy(self):
        '''
        Update the current policy based off of
        recently evaluating state values
        :return:
        '''

        for state in range(1,11):

            # if state value > 0 then the action is better than folding
            if self.policy_value[state] > 0:
                self.policy[state] = 0

            else:
                self.policy[state] = 1



# Lets build out our Deep RL player we are going to use the DQN model as described in Deepminds
# paper https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf
class DeepPlayer:
    # simple DQN agent
    def __init__(self, state_size, action_size):

        self.state_size = state_size
        self.actions = {0:'HIT' ,1:'STAY'}
        self.action_size = action_size
        self.feature_layer = None
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 0.65  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense( 4 *self.state_size, input_dim=self.state_size, activation='relu',use_bias=False))
        model.add(Dense( 4 *self.state_size, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse' ,optimizer=Adam(lr=self.learning_rate))
        return model

    def create_policy_df(self):


        policy_array = []
        # loop over dealer up card
        for i in range(1,11):
            # loop over possible player totals
            for j in range(4,22):
                # whether player is holding playable ace
                for k in range(2):

                    state = np.array([[i,j,k]])

                    action = self.make_max_play(state)

                    policy_array.append([i,j,k,action])


        policy_df = pd.DataFrame(policy_array,columns=['Dealer','Player','Ace','Policy'])

        policy_df.to_pickle('/Users/befeltingu/DeepRL/Data/DataFrames/blackjack_policy0')

    def create_similar_minibatch(self,minibatch):
        pass


    def make_play(self, state):

        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def make_max_play(self,state):

        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def get_action_number(self ,action_string):

        for key,value in self.actions.iteritems():

            if action_string == value:
                return key

        print("Error in get_action_number incorrect action string {}".format(action_string))

    def get_player_state(self ,cards):

        usable_ace = 0
        sum_cards = np.array(cards).sum()

        if 1 in cards:
            sum_cards_use_ace = sum_cards + 10
            if sum_cards_use_ace <= 21:
                usable_ace = 1
                sum_cards = sum_cards_use_ace

        return sum_cards, usable_ace

    def get_player_score(self,cards):

        sum_cards = np.array(cards).sum()

        if 1 in cards:
            sum_cards_use_ace = sum_cards + 10
            if sum_cards_use_ace <= 21:
                sum_cards = sum_cards_use_ace

        return sum_cards

    def load(self, name):
        self.model.load_weights(name)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)

        for state, action, reward, next_state, done in minibatch:

            target = reward

            if not done:
                target = (reward + self.gamma *
                          np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)

            target_f[0][action] = target

            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def replay_similarity(self, batch_size):

        minibatch = random.sample(self.memory, batch_size)

        for state, action, reward, next_state, done in minibatch:

            target = reward

            if not done:

                target = (reward + self.gamma *
                          np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)

            target_f[0][action] = target

            self.model.fit(state, target_f, epochs=1, verbose=0)

            # using the same reward get a bunch of similar states and fit the model with the expectation that
            # similar states would lead to similar outcomes especailly in the case when the episode is finished.

            state_feature_vector = np.matmul(state, self.feature_layer)

            similar_states = get_similar_vectors(state,state_feature_vector,self.feature_layer,250,1)

            if done:
                for similar_state in similar_states:

                    similar_state = np.reshape(similar_state,(1,3))

                    target_f = self.model.predict(similar_state)

                    target_f[0][action] = target

                    self.model.fit(similar_state, target_f, epochs=1, verbose=0)



        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self, name):
        self.model.save_weights(name)

