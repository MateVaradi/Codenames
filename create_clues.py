"""
Using W2V embeddings for automatic clue generation for Code Names.

"""
# Imports
import gensim
import time
import numpy as np
from functools import reduce
from operator import iconcat
from itertools import combinations
import os

# Settings
np.set_printoptions(suppress=True)


class CNbot:

    # Init
    def __init__(self, starting_board, our_color,
                 model_version='ft_1'):
        self.starting_board = starting_board
        self.current_board = starting_board
        self.our_color = our_color
        if self.our_color == "red":
            self.their_color = "blue"
        elif self.our_color == "blue":
            self.their_color = "red"

        # Define lists of words of the starting board and the current that we do not have (or had) to guess:
        self.y1_t = self.current_board[self.their_color]  # opponent's card
        self.y1_0 = self.starting_board[self.their_color]
        self.y2_t = self.current_board["black"]  # black cards
        self.y2_0 = self.starting_board["black"]
        self.y3_t = self.current_board["beige"]  # beige cards
        self.y3_0 = self.starting_board["beige"]
        self.y_0 = self.y1_0 + [self.y2_0] + self.y3_0
        self.y_t = self.y1_t + [self.y2_t] + self.y3_t
        self.num_y1_t = len(self.y1_t)
        self.num_y2_t = len(self.y2_t)
        self.num_y3_t = len(self.y3_t)
        self.num_y_t = self.num_y3_t + self.num_y3_t + self.num_y3_t  # number of cards we don't have to guess

        # Define lists of words of the starting board and the current that we do have (or had) to guess:
        self.x_t = self.current_board[self.our_color]  # remaining words that our team has to guess
        self.x_0 = self.starting_board[self.our_color]  # remaining words that our team has to guess
        self.num_x_t = len(self.x_t)  # number of words our team has to guess

        self.all_cards = self.x_t + self.y_t

        # Load embeddings
        if not os.path.exists('embeddings/' + model_version):
            model_version = self.get_embeddings(model_version)

        self.model = self.load_model(model_version)
        self.model_vocab = sorted(self.model.vocab.keys())

    def load_model(self, model_version):
        fasttext_model = gensim.models.Word2Vec.load('embeddings/' + model_version)
        w2v = fasttext_model.wv
        del fasttext_model
        return w2v

    def update_board(self, new_board):
        self.current_board = new_board
        self.y1_t = self.current_board[self.their_color]  # opponent's card
        self.y2_t = self.current_board["black"]  # black cards
        self.y3_t = self.current_board["beige"]  # beige cards
        self.y_t = self.y1_t + [self.y2_t] + self.y3_t
        self.num_y1_t = len(self.y1_t)
        self.num_y2_t = len(self.y2_t)
        self.num_y3_t = len(self.y3_t)
        self.num_y_t = self.num_y3_t + self.num_y3_t + self.num_y3_t  # number of cards we don't have to guess
        self.x_t = self.current_board[self.our_color]  # remaining words that our team has to guess
        self.num_x_t = len(self.x_t)  # number of words our team has to guess
        self.all_cards = self.x_t + self.y_t

    def show_current_game(self):
        print('We (team ', self.our_color, ') have ', self.num_x_t, ' cards left to guess.')
        print('Team ', self.their_color, ' has ', self.num_y1_t, 'cards left')
        print('\n')
        print('The cards left are:')
        print('blue:', self.x_t)
        print('red:', self.y1_t)
        print('black:', self.y2_t)
        print('beige:', self.y3_t)
        print('\n')

    def get_clues(self, method, method_params, num_clues=10, max_m=4):

        if method == "ST":
            if method_params:
                self.ST_clues(method_params=method_params, num_clues=num_clues, max_m=max_m)
            else:
                self.ST_clues(num_clues=num_clues, max_m=max_m)
        elif method == "WSD":
            if (not hasattr(self, 'X')) or (not hasattr(self, 'Y')):
                self.load_similarity_matrix()
            if method_params:
                self.WSD_clues(method_params=method_params, num_clues=num_clues, max_m=max_m)
            else:
                self.WSD_clues(num_clues=num_clues, max_m=max_m)
        else:
            print('Method should be either "ST" or "WSD"')

    def load_similarity_matrix(self):
        X = self.create_similarity_matrix(self.x_0)  # word similarities to 'good' words
        Y = self.create_similarity_matrix(self.y_0)  # word similarities to 'bad' words
        self.X = X
        self.Y = Y

    def create_similarity_matrix(self, card_list):
        """
        Fill and return v x n matrix of similarities between words of the cards and the vocabulary
        Inputs:
            card_list : list of cards
        """
        n = len(card_list)  # number of cards
        v = len(self.model_vocab)  # number of words in vocab

        S = np.zeros((v, n))
        for i, word in enumerate(self.model_vocab):
            for j, card in enumerate(card_list):
                S[i, j] = self.model.similarity(word, card)

        return S

    # Similarity Threshold method
    def ST_clues(self, num_clues, max_m, method_params=(0.3, 0.15, 0.8, 0.2)):
        """
        Prints num_clues word clues, the number of cards for each clue

        Inputs:
            num_clues       : number of clues to print
            max_m           : maximum number of cards to cover with clue
            method_params:
                epsilon_1   : similarity threshold for opponents cards
                epsilon_2   : similarity threshold for black card
                epsilon_3   : similarity threshold for beige cards
                r           : reward for larger m
        """
        t0 = time.time()
        print('Thinking about clues for the ', self.our_color, ' team')

        # Preparations
        epsilon_1, epsilon_2, epsilon_3, r = method_params
        max_m = min(self.num_x_t, max_m)  # maximal clue number
        min_m = 2
        poss_card_combos = reduce(iconcat, [list(combinations(self.x_t, a)) for a in range(min_m, max_m + 1)], [])

        # Initialize lists to store results
        best_clues = ['']
        best_sims = [-np.inf]
        ms = [0]
        already_tried = []

        # Loop through all possible card combinations
        for poss_card_combo in poss_card_combos:
            poss_card_combo = list(poss_card_combo)
            m = len(poss_card_combo)
            clue_candidates = self.model.most_similar(poss_card_combo, topn=30)
            # Exclude clues that contain one of the cards of the board
            clue_candidates = [elem for elem in clue_candidates if
                               not any((elem[0] in card) or (card in elem[0]) for card in self.all_cards)]
            # Exclude clues where similarity with black or red cards is high
            for elem in clue_candidates:
                clue_candidate = elem[0]
                if not clue_candidate in already_tried:
                    sim_candidate = elem[1] + r * (m - 2)
                    opponent_similarities = [self.model.similarity(clue_candidate, opponent_card) > epsilon_1
                                             for opponent_card in self.y1_t]
                    black_similarity = self.model.similarity(clue_candidate, self.y2_t) > epsilon_2
                    beige_similarities = [self.model.similarity(clue_candidate, beige_card) > epsilon_3
                                          for beige_card in self.y3_t]
                    # We save the clue if it is not too similar to the cards we don't want to guess
                    if (not any(opponent_similarities)) and (not black_similarity) and \
                            (not any(beige_similarities)) and (not clue_candidate in best_clues):
                        best_clues.append(clue_candidate)
                        best_sims.append(sim_candidate)
                        ms.append(m)
                    else:
                        already_tried.append(clue_candidate)
        # Save best num_clues clues
        top_clues_ind = np.argsort(best_sims)[-(num_clues + 1):-1][::-1]
        best_clues = np.array(best_clues)[list(top_clues_ind)]
        best_sims = np.array(best_sims)[list(top_clues_ind)]
        best_ms = np.array(ms)[list(top_clues_ind)]

        for i, elem in enumerate(zip(best_clues, best_ms, best_sims)):
            print(i + 1, ") clue:", elem[0],
                  ", number of cards (m):", elem[1],
                  ", score:", round(elem[2], 2))

        print('Clue generation took', round(time.time() - t0, 1), ' seconds.')
        print('\n')

    def match_clue_candidate(self, a, poss_idx, r=0.05):
        """
        Returns the score corresponding to the best combination along with the number of cards included
        Inputs:
                a: a column of the similarity matrix S
                poss_indices: all possible index combinations
                r: factor that rewards when more cards are used.
                    r=0 means there is no reward for a clue that maps to more cards.
        """
        # Initialize values
        s = -np.inf  # score
        m = 0  # number of words for clue

        for poss_id in poss_idx:
            poss_m = len(poss_id)
            poss_score = sum(a[list(poss_id)]) / poss_m + (poss_m - 2) * r
            if poss_score > s:
                s = poss_score
                m = poss_m

        return (s, m)

    # Weighted Similarity Difference method
    def WSD_clues(self, num_clues, max_m, method_params=(1, 1.4, 0.3, 0.05)):
        """
        Prints num_clues word clues, the number of cards for each clue and the margins

        Inputs:
            num_clues   : number of clues to print
            max_m       : maximum number of cards to cover with clue
            method_params:
                theta_1     : weight for opponents cards
                theta_2     : weight for black card
                theta_3     : weight for beige cards
                r           : factor that rewards when more cards are used.
                            r=0 means there is no reward for covering more cards.
        """
        t0 = time.time()
        print('Thinking about clues for the ', self.our_color, ' team')

        # Preparations
        theta_1, theta_2, theta_3, r = method_params
        x_t_ind = [self.x_0.index(word) for word in self.x_t]  # row indices (in X) corresponding to our cards
        y1_ind = [self.y_0.index(word) for word in self.y1_t]  # row indices (in Y) corresponding to opponent's cards
        y2_ind = self.y_0.index(self.y2_t)  # row index corresponding (in Y) to the black card
        y3_ind = [self.y_0.index(word) for word in self.y3_t]  # row indices (in Y) corresponding to beige cards
        max_m = min(self.num_x_t, max_m)  # maximal clue number
        min_m = 2
        poss_indices = reduce(iconcat, [list(combinations(x_t_ind, m)) for m in range(min_m, max_m + 1)], [])

        # Calculations
        weights = np.array([theta_1] * self.num_y1_t + [theta_2] + [theta_3] * self.num_y3_t)
        beta_and_clue_nums = np.apply_along_axis(lambda a: self.match_clue_candidate(a, poss_indices, r), 1, self.X)
        gamma = np.apply_along_axis(lambda a: np.sum(a[y1_ind + [y2_ind] + y3_ind] * weights / self.num_y_t), 1, self.Y)
        beta = beta_and_clue_nums[:, 0]
        clue_nums = beta_and_clue_nums[:, 1]
        del beta_and_clue_nums
        delta = beta - gamma

        # Print clues
        i = 1
        clue_ind = 0
        while clue_ind < num_clues - 1:
            next_best_ind = delta.argsort()[-i]
            clue = self.model_vocab[next_best_ind]
            # Do not consider words from the cards of board (or part of the cards) as clues
            if not any((clue in card) or (card in clue) for card in self.all_cards):
                print(clue_ind + 1, ") clue:", clue,
                      ", number of cards (m): ", round(clue_nums[next_best_ind]),
                      ", score (margin) : ", round(delta[next_best_ind], 2))
                clue_ind += 1
            i += 1

        print('Clue generation took', round(time.time() - t0, 1), ' seconds.')
        print('\n')


# Example run

# Initial board example
board_0 = {'blue': ['bank', 'limousine', 'plastic', 'stadium', 'drill', 'moon', 'stick', 'forest', 'china'],
           'red': ['nut', 'hotel', 'berlin', 'slug', 'dog', 'cliff', 'block', 'fish'],
           'black': 'tokyo',
           'beige': ['horse', 'dinosaur', 'ninja', 'plate', 'button', 'tube', 'spot']}

# Board after a few rounds
board_t = {
    'blue': ['plastic', 'stadium', 'drill', 'moon', 'stick', 'forest', ],
    'red': ['nut', 'hotel', 'dog', 'block', 'fish'],
    'black': 'tokyo',
    'beige': ['horse', 'dinosaur', 'ninja', 'button', 'tube', 'stadium']
}

# Initialize Codenames bot
game = CNbot(board_0, our_color='blue')
game.show_current_game()

# Try both methods to get clues:
# 1) Weighted Similarity Distance
game.get_clues(method='WSD', num_clues=5, max_m=4, method_params=(1, 1.4, 0.8, 0.05))
# 2) Similarity Threshold
game.get_clues(method='ST', num_clues=5, max_m=4, method_params=(0.3, 0.15, 0.8, 0.2))

# Update board after some cards have been quessed
game.update_board(board_t)
game.show_current_game()

###

board_3 = {'blue':['copper','ivory','compound','beer','vet','pirate','drill','pumpkin'],
           'red' : ['fair','superior','litter','wash','olive','cycle','paste','china','lead'],
            'black' : 'air',
           'beige' : ['air','oil','england','ring','point','slip','duck']
}

## -------------------------------------------------------------------------------------
## Playground for you:
board_2 = {
    'blue': ['chicken', 'pork', 'beef', 'olympus',
             'player', 'drill', 'apple', 'strawberry', 'peach'],
    'red': ['cat', 'nut', 'berlin', 'cliff',
            'hotel', 'dog', 'block', 'fish'],
    'black': 'tokyo',
    'beige': ['horse', 'dinosaur', 'ninja', 'plate', 'button',
              'tube', 'stadium']
}
