# -*- encoding: utf-8 -*-

import numpy as np
import random



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

class Dealer:

    def __init__(self,decks):
        # integer representing the number of decks the dealer has
        self.decks = decks
        self.shoe = []
        self.showing_card = None
        self.current_score = None

    def deal_card(self):

        draw_card = random.choice(self.shoe)

        self.shoe.remove(draw_card)

        return draw_card

    def deal_and_replace(self):

        rand_card = np.random.randint(1,14)
        if rand_card >= 10:
            return 10
        else:
            return rand_card
    # pass in deck and return random sorted deck
    def shuffle(self,deck):

        old_deck = deck
        new_deck = []
        while(len(old_deck) > 0):

            draw = random.choice(old_deck)
            new_deck.append(draw)
            old_deck.remove(draw)

        return new_deck

    # either set the shoe for the first time or reset it
    def set_shoe(self):

        clean_shoe = []
        for di in range(self.decks):
            # dont have to worry about suites for blackjack
            for i in range(1,5):
                # A = 1
                # 2 = 2
                # ....
                for j in range(1,14):
                    clean_shoe.append(j)


        shuffle_shoe = self.shuffle(clean_shoe)
        self.shoe = shuffle_shoe

    def make_play(self,cards):

        sum_cards = np.array(cards).sum()

        if sum_cards == 2:
            return "HIT"

        if 1 in cards:
            sum_cards_use_ace = sum_cards + 10
            if sum_cards_use_ace <= 21:
                sum_cards = sum_cards_use_ace

        self.current_score = sum_cards
        if sum_cards < 17:
            return "HIT"

        elif sum_cards > 21:

            return "BUST"

        else:

            return "STAY"


def simulate_simple_game():

    '''
    Using this game to demonstrate policy iteration

    The game is played different from regular blackjack
    first the dealer is dealt one up card and one downcard
    The player first antes then desides based off of what
    the dealer is showing to place a bet or not. The bet size
    is fixed before hand so the player cant just bet more the
    more of an advantage he has

    1: init policy :

    2: policy evaluation :

    3: policy improvement :

    :return:
    '''

    # STEP 1 INIT STEPS
    dealer = Dealer(1)

    ante_size = 1
    bet_size = 10
    eval_iters = 1000

    player = PolicyIterPlayer(ante_size=ante_size,bet_size=bet_size,eval_iters=eval_iters,dealer=dealer)

    player.init_policy()

    for episode in range(15):

        player.evaluate_policy_0()

        player.improve_policy()


    return player

def simulate_simple_game2():

    '''
    same as simple game except now the player
    attempts to make 'good' plays
    :return:
    '''

    # STEP 1 INIT STEPS
    dealer = Dealer(1)

    ante_size = 1
    bet_size = 10
    eval_iters = 100000

    player = PolicyIterPlayer(ante_size=ante_size, bet_size=bet_size, eval_iters=eval_iters, dealer=dealer)

    player.init_policy()

    for episode in range(5):
        player.evaluate_policy_1()

        player.improve_policy()

    return player



if __name__ == '__main__':

    import json
    player = simulate_simple_game2()


    print("Done running policy iteration for simple blackjack")

    print("Current value function")

    print(json.dumps(player.policy_value))

    print("Current policy")

    print(json.dumps(player.policy))


