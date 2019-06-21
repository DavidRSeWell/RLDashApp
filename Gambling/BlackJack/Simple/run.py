import numpy as np

from player import DeepPlayer
from dealer import Dealer
from keras import backend as K



def get_avg_score(player, dealer, num_episodes):
    '''
    Run a number of hands and track the avg return over those hands
    :param player:
    :param dealer:
    :param num_episodes:
    :return:
    '''

    tot_reward = 0

    for i in range(num_episodes):

        # player cards
        pc1 = dealer.deal_and_replace()
        pc2 = dealer.deal_and_replace()
        player_cards = [pc1, pc2]
        # dealer cards
        dc1 = dealer.deal_and_replace()  # Use this as the visible card
        dc2 = dealer.deal_and_replace()
        dealer_cards = [dc1, dc2]

        game_state = "HIT"
        # deal cards until the player says Stay
        while (game_state == "HIT"):

            player_sum, usable_ace = player.get_player_state(player_cards)

            current_state = np.array([[dc1, player_sum, usable_ace]])

            player_action = player.make_max_play(current_state)

            action_type = player.actions[player_action]

            if action_type == "STAY":
                game_state = "STAY"
                break

            else:  # player hit

                new_card = dealer.deal_and_replace()
                player_cards += [new_card]

                if sum(player_cards) > 21:
                    game_state = "BUST"
                    break

        if game_state == "BUST":
            tot_reward -= 1  # Player busted
            continue

        # now do the same loop for the dealer
        # the dealer is using a different policy
        dealer_state = "HIT"
        while (dealer_state == "HIT"):

            dealer_state = dealer.make_play(dealer_cards)

            if dealer_state == "STAY":
                break

            elif dealer_state == "BUST":
                break

            else:
                new_card = dealer.deal_and_replace()
                dealer_cards += [new_card]

        if dealer_state == "BUST":
            # dealer busted so record the current state as a win
            tot_reward += 1
            continue

        player_score = player.get_player_score(player_cards)
        dealer_score = dealer.current_score

        reward = 0

        if player_score > dealer_score:
            tot_reward += 1
        elif dealer_score > player_score:
            tot_reward -= 1

    return tot_reward / float(num_episodes)

def run_black_jack(player, dealer, batch_size, num_episodes, game_sample_size, avg_score_sample_size,player_type="normal"):
    ####### LOOP #####################
    # a.) generate an episode (single deal)
    # b.) For each state s in episode append returned value
    # to Returns(s)

    # INIT STEPS
    # number of games/episodes to play

    game_sample_size = game_sample_size  # number hands to deal to given policy before updating
    episodes = num_episodes
    reward_episode = []

    for episode in range(episodes):

        for i in range(game_sample_size):

            # player cards
            pc1 = dealer.deal_and_replace()
            pc2 = dealer.deal_and_replace()
            player_cards = [pc1, pc2]
            # dealer cards
            dc1 = dealer.deal_and_replace()  # Use this as the visible card
            dc2 = dealer.deal_and_replace()
            dealer_cards = [dc1, dc2]

            game_state = "HIT"
            # deal cards until the player says Stay
            while (game_state == "HIT"):

                player_sum, usable_ace = player.get_player_state(player_cards)

                current_state = np.array([[dc1, player_sum, usable_ace]])

                player_action = player.make_play(current_state)

                action_type = player.actions[player_action]

                if action_type == "STAY":
                    game_state = "STAY"
                    break

                else:  # player hit

                    new_card = dealer.deal_and_replace()
                    player_cards += [new_card]

                    if sum(player_cards) > 21:
                        game_state = "BUST"
                        break

                    reward = 0  # no actual reward yet

                    player_new_sum, usable_ace = player.get_player_state(player_cards)

                    next_state = np.array([[dc1, player_new_sum, usable_ace]])
                    done = False
                    player.remember(current_state, player_action, reward, next_state, done)

            if game_state == "BUST":
                player_sum, usable_ace = player.get_player_state(player_cards)
                current_state = np.array([[dc1, player_sum, usable_ace]])
                action = 0  # the player hit
                reward = -1  # We lose :(
                next_state = np.zeros((1, 3))  # no next step episode is over
                done = True
                player.remember(current_state, action, reward, next_state, done)
                continue
            # now do the same loop for the dealer
            # the dealer is using a different policy
            dealer_state = "HIT"
            while (dealer_state == "HIT"):

                dealer_state = dealer.make_play(dealer_cards)

                if dealer_state == "STAY":
                    break

                elif dealer_state == "BUST":
                    break

                else:
                    new_card = dealer.deal_and_replace()
                    dealer_cards += [new_card]

            if dealer_state == "BUST":
                # dealer busted so record the current state as a win
                player_sum, usable_ace = player.get_player_state(player_cards)
                current_state = np.array([[dc1, player_sum, usable_ace]])
                action = player.get_action_number(game_state)  # could have hit or stayed to have got here
                reward = 1  # we won!
                next_state = np.zeros((1, 3))  # no next step episode is over
                done = True
                player.remember(current_state, action, reward, next_state, done)
                continue

            player_score = player.get_player_score(player_cards)
            dealer_score = dealer.current_score

            reward = 0

            if player_score > dealer_score:
                reward = 1
            elif dealer_score > player_score:
                reward = -1

            player_sum, usable_ace = player.get_player_state(player_cards)
            current_state = np.array([[dc1, player_sum, usable_ace]])
            action = player.get_action_number(game_state)  # could have hit or stayed to have got here
            next_state = np.zeros((1, 3))  # no next step episode is over
            done = True
            player.remember(current_state, action, reward, next_state, done)

        # finished current hand sample
        if len(player.memory) > batch_size:

            if player_type == "normal":
                player.replay(batch_size)
            elif player_type == "similar":
                player.replay_similarity(batch_size)
            else:
                print("Errror wrong player type specified")


        if (episode % 50) == 0:

            print("Finished episode {episode} Epsilon= {epsilon}".format(episode=episode, epsilon=player.epsilon))

            avg_reward = get_avg_score(player,dealer,num_episodes=avg_score_sample_size)

            print("Avg Reward : {}".format(avg_reward))

            reward_episode.append(avg_reward)


    # Done running save off avg rewards per episode

    avg_reward = np.array(reward_episode)

    if player_type == 'similar':

        np.save('/Users/befeltingu/DeepRL/Data/Graph/black_jack_reward_similar.npy', avg_reward)

    else:

        np.save('/Users/befeltingu/DeepRL/Data/Graph/black_jack_reward.npy', avg_reward)


if __name__ == '__main__':


    run_normal_player = 0
    if run_normal_player:

        print("Running normal player")

        player = DeepPlayer(3, 2)

        dealer = Dealer(1)

        run_black_jack(player, dealer, batch_size=32, num_episodes=1000, game_sample_size=1000, avg_score_sample_size=5000,player_type="normal")

        player.create_policy_df()

        player.save('/Users/befeltingu/DeepRL/Data/models/blackjack_basic_model.h5')

    run_similar_player = 1
    if run_similar_player:

        print("Running similarity player")
        player = DeepPlayer(3, 2)

        player.feature_layer = K.eval(player.model.layers[0].weights[0])

        dealer = Dealer(1)

        run_black_jack(player, dealer, batch_size=32, num_episodes=1000, game_sample_size=1000,
                       avg_score_sample_size=5000, player_type="similar")

        player.create_policy_df()

        player.save('/Users/befeltingu/DeepRL/Data/models/blackjack_basic_similarity_model_1.h5')
