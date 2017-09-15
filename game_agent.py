"""Finish all TODO items in this file to complete the isolation project, then
test your agent's strength against a set of known agents using tournament.py
and include the results in your report.
"""
import random


class SearchTimeout(Exception):
    """Subclass base exception for code clarity. """
    pass


def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    This should be the best heuristic function for your project submission.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    score_2 = custom_score_2(game, player)
    score_3 = custom_score_3(game, player)


    # Heuristic chosen for custom_score function by multiplying previous scores together
    return 0.35 * score_2 + 0.65 * score_3


def custom_score_2(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    # TODO: finish this function!
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    own_moves = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))

    # multiply opponent moves by own moves - 3:
    return float(own_moves - 3 * opp_moves)


def custom_score_3(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """

    # helper function to identify if a move is near a border of the isolation board
    def approaching_border(move, borders):
        for border in borders:
            if move in border:
                return True
        return False


    # helper function identifies the percent of board filled in current game state
    def percent_of_board_filled(game):
        blank_spaces = game.get_blank_spaces()
        return int((len(blank_spaces) / (game.width * game.height)) * 100)

    borders = [
        [(0, i) for i in range(game.width)],
        [(i, 0) for i in range(game.height)],
        [(game.width - 1, i) for i in range(game.width)],
        [(i, game.height - 1) for i in range(game.height)]
    ]

    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    agent_moves = game.get_legal_moves(player)
    opponent_moves = game.get_legal_moves(game.get_opponent(player))

    agent_score = 0
    opponent_score = 0

    # assign a value to the current player'ss score based on percent of board filled
    for move in agent_moves:
        if percent_of_board_filled(game) < 30:
            agent_score += 10
        elif 30 > percent_of_board_filled(game) < 85 and approaching_border(move, borders):
            agent_score -= 30
        elif percent_of_board_filled(game) > 85 and approaching_border(move, borders):
            agent_score -= 40
        elif not approaching_border(move, borders):
            agent_score += 10

    # assign a value to the opponent's score based on percent of board filled
    for move in opponent_moves:
        if percent_of_board_filled(game) < 30:
            opponent_score += 10
        elif 30 > percent_of_board_filled(game) < 85 and approaching_border(move, borders):
            opponent_score -= 30
        elif percent_of_board_filled(game) > 85 and approaching_border(move, borders):
            opponent_score -= 40
        elif not approaching_border(move, borders):
            opponent_score += 10

    # function subtracts the opponent's final score from the player's score to get the heuristic value
    return float(agent_score - opponent_score)


class IsolationPlayer:
    """Base class for minimax and alphabeta agents -- this class is never
    constructed or tested directly.

    ********************  DO NOT MODIFY THIS CLASS  ********************

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """
    def __init__(self, search_depth=3, score_fn=custom_score, timeout=10.):
        self.search_depth = search_depth
        self.score = score_fn
        self.time_left = None
        self.TIMER_THRESHOLD = timeout


class MinimaxPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using depth-limited minimax
    search. You must finish and test this player to make sure it properly uses
    minimax to return a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        **************  YOU DO NOT NEED TO MODIFY THIS FUNCTION  *************

        For fixed-depth search, this function simply wraps the call to the
        minimax method, but this method provides a common interface for all
        Isolation agents, and you will replace it in the AlphaBetaPlayer with
        iterative deepening search.

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
        best_move = (-1, -1)

        try:
            # The try/except block will automatically catch the exception
            # raised when the timer is about to expire.
            return self.minimax(game, self.search_depth)

        except SearchTimeout:
            return best_move # Handle any actions required after timeout as needed

        # Return the best move from the last completed search iteration
        return best_move

    def minimax(self, game, depth):
        """Implement depth-limited minimax search algorithm as described in
        the lectures.

        This should be a modified version of MINIMAX-DECISION in the AIMA text.
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Minimax-Decision.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """

        # (based on AIMA text - minimax contains functions vax and min value which call each other recursively


        def max_val(game, depth):
            # check the time left
            if self.time_left() < self.TIMER_THRESHOLD:
                raise SearchTimeout()

            if depth == 0 or len(game.get_legal_moves()) == 0:
                return self.score(game, self)

            legal_moves = game.get_legal_moves()
            main_score = float('-inf')

            # for each available move in the game tree, calculate the minimax value
            for each_move in legal_moves:
                main_score = max(main_score,min_val(game.forecast_move(each_move),depth-1))
            return main_score

        def min_val(game, depth):
            # check the time left
            if self.time_left() < self.TIMER_THRESHOLD:
                raise SearchTimeout()

            if depth == 0 or len(game.get_legal_moves()) == 0:
                return self.score(game, self)

            legal_moves = game.get_legal_moves()
            main_score = float('inf')

            # for each available move in the game tree, calculate the minimax value
            for each_move in legal_moves:
                main_score = min(main_score,max_val(game.forecast_move(each_move),depth-1))
            return main_score

        legal_moves = game.get_legal_moves()

        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        #if there are no legal moves then return the default move (-1, -1)
        if not legal_moves:
            return -1, -1

        # initiate search of game tree
        best_move = legal_moves[0]
        best_score = float("-inf")
        v = float("-inf")
        for move in legal_moves:
            v = max(min_val(game.forecast_move(move), depth-1), v)
            if v > best_score:
                best_score = v
                best_move = move
        return best_move


class AlphaBetaPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using iterative deepening minimax
    search with alpha-beta pruning. You must finish and test this player to
    make sure it returns a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        Modify the get_move() method from the MinimaxPlayer class to implement
        iterative deepening search instead of fixed-depth search.

        **********************************************************************
        NOTE: If time_left() < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout

        best_move = (-1, -1)
        try:
            iterative_depth = 1
            while True:
                best_move = self.alphabeta(game, iterative_depth)
                iterative_depth += 1
        except SearchTimeout:
            return best_move
        return best_move

    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf")):
        """Implement depth-limited minimax search with alpha-beta pruning as
        described in the lectures.

        This should be a modified version of ALPHA-BETA-SEARCH in the AIMA text
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Alpha-Beta-Search.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        # min and max functions call each other recursively
        def max_val(game, depth, alpha, beta):
            # check time left
            if self.time_left() < self.TIMER_THRESHOLD:
                raise SearchTimeout()
            # if node is terminal or no moves available return score
            if depth == 0 or len(game.get_legal_moves()) == 0:
                return self.score(game, self)

            legal_moves = game.get_legal_moves()
            main_score = float('-inf')

            for each_move in legal_moves:
                # for each available move calculate the minimax
                main_score = max(main_score, min_val(game.forecast_move(each_move), depth-1, alpha, beta))
                # check if the current main score is >= than beta, if so prune
                if main_score >= beta:
                    return main_score
                # if main score < beta keep searching
                alpha = max(main_score, alpha)
            return main_score

        def min_val(game, depth, alpha, beta):
            # check time left
            if self.time_left() < self.TIMER_THRESHOLD:
                raise SearchTimeout()
            # if node is terminal or no moves available return score
            if depth == 0 or len(game.get_legal_moves()) == 0:
                return self.score(game, self)

            legal_moves = game.get_legal_moves()
            main_score = float('inf')

            for each_move in legal_moves:
                # for each available move calculate the minimax
                main_score = min(main_score, max_val(game.forecast_move(each_move), depth-1, alpha, beta))
                # check if main score <=alpha, if so then prune
                if main_score <= alpha:
                    return main_score
                # if main score > alpha keep searching
                beta = min(main_score, beta)
            return main_score

        legal_moves = game.get_legal_moves()

        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        # if there are no legal moves:
        if not legal_moves:
            return -1, -1

        best_move = legal_moves[random.randint(0, len(legal_moves)-1)]
        best_score = float("-inf")
        v = float("-inf")

        # initiate search of game tree
        for move in legal_moves:
            v = max(min_val(game.forecast_move(move), depth-1, alpha, beta), v)
            if best_score < v:
                best_score = v
                best_move = move
                if v >= beta:
                    return best_move
            alpha = max(v, alpha)
        return best_move
