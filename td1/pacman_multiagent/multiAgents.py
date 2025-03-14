# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import random, util

from game import Agent


class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """

    def get_action(self, game_state):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legal_moves = game_state.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluation_function(game_state, action) for action in legal_moves]
        best_score = max(scores)
        best_indices = [index for index in range(len(scores)) if scores[index] == best_score]
        chosen_index = random.choice(best_indices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legal_moves[chosen_index]

    def evaluation_function(self, current_game_state, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (new_food) and Pacman position after moving (new_pos).
        new_scared_times holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successor_game_state = current_game_state.generatePacmanSuccessor(action)
        new_pos = successor_game_state.getPacmanPosition()
        new_food = successor_game_state.getFood()
        new_ghost_states = successor_game_state.getGhostStates()
        new_scared_times = [ghostState.scaredTimer for ghostState in new_ghost_states]
        walls = current_game_state.getWalls()
        score = successor_game_state.getScore()

        if walls[new_pos[0]][new_pos[1]]:
            score = float('-inf')
        if action == 'Stop':
            score -= 500
        # Ajout d'une récompense pour se rapprocher de la nourriture
        food_list = new_food.asList()
        if food_list:
            min_food_distance = min(util.manhattan_distance(new_pos, food) for food in food_list)
            score += 10 / (min_food_distance + 1)  # Plus proche de la nourriture = meilleur score

        # Gestion des fantômes (éviter ou attaquer s'ils sont effrayés)
        for i, state in enumerate(new_ghost_states):
            ghost_distance = util.manhattan_distance(state.getPosition(), new_pos)
            if new_scared_times[i] > 0:
                # Récompense Pac-Man s'il va vers un fantôme effrayé
                score += 200 / (ghost_distance + 1)
            elif ghost_distance < 3:
                # Pénalité plus forte si un fantôme dangereux est proche
                score -= (200 / (ghost_distance + 1))

        return score


def score_evaluation_function(current_game_state):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search game
      (not reflex game).
    """
    return current_game_state.getScore()


class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search game.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn='score_evaluation_function', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def get_action(self, game_state):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of game in the game
        """
        "*** YOUR CODE HERE ***"

        def minimax(agent_index, depth, state):
            # If max depth reached or game over, return evaluation
            if depth == self.depth or state.isWin() or state.isLose():
                return self.evaluationFunction(state)

            # Pac-Man's turn (Maximizing)
            if agent_index == 0:
                best_value = float('-inf')
                best_action = None
                for action in state.getLegalActions(agent_index):
                    successor = state.generateSuccessor(agent_index, action)
                    value = minimax(1, depth, successor)  # Move to next agent
                    if value > best_value:
                        best_value = value
                        best_action = action
                return best_action if depth == 0 else best_value

            # Ghosts' turn (Minimizing)
            else:
                next_agent = agent_index + 1
                if next_agent == state.getNumAgents():  # Last ghost, go to next depth
                    next_agent = 0
                    depth += 1
                return min(minimax(next_agent, depth, state.generateSuccessor(agent_index, action))
                           for action in state.getLegalActions(agent_index))

        return minimax(0, 0, game_state)


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def get_action(self, game_state):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        def alphabeta(agent_index, depth, state, alpha, beta):
            # Condition d'arrêt (profondeur max ou état terminal)
            if depth == self.depth or state.isWin() or state.isLose():
                return self.evaluationFunction(state)

            legal_actions = state.getLegalActions(agent_index)
            if not legal_actions:  # Si aucun coup légal, retourner l'évaluation de l'état
                return self.evaluationFunction(state)

            # Tour de Pac-Man (Maximisation)
            if agent_index == 0:
                best_value = float('-inf')
                best_action = None
                for action in legal_actions:
                    successor = state.generateSuccessor(agent_index, action)
                    value = alphabeta(1, depth, successor, alpha, beta)  # Tour du premier fantôme
                    if value > best_value:
                        best_value = value
                        best_action = action
                    alpha = max(alpha, best_value)
                    if beta <= alpha:  # Élagage (on arrête d'explorer cette branche)
                        break
                return best_action if depth == 0 else best_value

            # Tour des Fantômes (Minimisation)
            else:
                next_agent = agent_index + 1
                if next_agent == state.getNumAgents():  # Dernier fantôme, passer au tour suivant
                    next_agent = 0
                    depth += 1
                best_value = float('inf')
                for action in legal_actions:
                    successor = state.generateSuccessor(agent_index, action)
                    value = alphabeta(next_agent, depth, successor, alpha, beta)
                    best_value = min(best_value, value)
                    beta = min(beta, best_value)
                    if beta <= alpha:  # Élagage
                        break
                return best_value

        return alphabeta(0, 0, game_state, float('-inf'), float('inf'))


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question  4)
    """

    def get_action(self, game_state):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"

        def expectimax(agent_index, depth, state):
            # Condition d'arrêt (profondeur max atteinte ou état terminal)
            if depth == self.depth or state.isWin() or state.isLose():
                return self.evaluationFunction(state)

            legal_actions = state.getLegalActions(agent_index)
            if not legal_actions:  # Si aucun mouvement possible, évaluer l'état
                return self.evaluationFunction(state)

            # Tour de Pac-Man (Maximisation)
            if agent_index == 0:
                best_value = float('-inf')
                best_action = None
                for action in legal_actions:
                    successor = state.generateSuccessor(agent_index, action)
                    value = expectimax(1, depth, successor)  # Tour du premier fantôme
                    if value > best_value:
                        best_value = value
                        best_action = action
                return best_action if depth == 0 else best_value

            # Tour des Fantômes (Valeur Expectée)
            else:
                next_agent = agent_index + 1
                if next_agent == state.getNumAgents():  # Dernier fantôme, avancer en profondeur
                    next_agent = 0
                    depth += 1
                values = [expectimax(next_agent, depth, state.generateSuccessor(agent_index, action))
                          for action in legal_actions]
                return sum(values) / len(values)  # Moyenne des valeurs (espérance)

        return expectimax(0, 0, game_state)


def betterEvaluationFunction(currentGameState):
    """
       evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raise_not_defined()


# Abbreviation
better = betterEvaluationFunction
