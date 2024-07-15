# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):

    stack = [(problem.getStartState(), [], set())] 
     #Start from the initial state obtained ,getStartState function returns x and y coordinates of the start state.

    while stack:
    # Initialize a stack to hold tuples of (state, actions) and a set visited to track visited states.
        state, actions, visited = stack.pop()
        
        if problem.isGoalState(state):
            return actions
        
        if state not in visited:
            visited.add(state)
            for successor, action, step_cost in problem.getSuccessors(state): 
                #Pop states from the stack, expand them by getting successors 
                if successor not in visited:
                    stack.append((successor, actions + [action], visited.copy()))
    
    return []
util.raiseNotDefined()

from collections import deque #deque (queue) to facilitate BFS, which operates in a FIFO manner.
#deque (double-ended queue) for BFS to efficiently append and pop elements from both ends.
def breadthFirstSearch(problem):
    queue = deque()
    visited = set()
    start_state = problem.getStartState()
    queue.append((start_state, []))  # (state, path)

    while queue:
        state, actions = queue.popleft()
#BFS explores each state level by level, ensuring that the shortest path (in terms of number of actions) is found first.
        if problem.isGoalState(state):
            return actions

        if state not in visited:
            visited.add(state)
            for successor, action, step_cost in problem.getSuccessors(state):
                if successor not in visited:
                    queue.append((successor, actions + [action]))

    return []  #first time encounter the goal state, we return the accumulated actions.
util.raiseNotDefined()


def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    priority_queue = []
    visited = set()
    start_state = problem.getStartState()
    priority_queue.append((0, (start_state, [])))  # (total_cost, (state, path))

    while priority_queue:
        priority_queue.sort(key=lambda x: x[0])  # Sort by total_cost
        total_cost, (state, actions) = priority_queue.pop(0)
#UCS explores nodes based on their cumulative path cost (total_cost).
        if problem.isGoalState(state):
            return actions

        if state not in visited:
            visited.add(state)
            for successor, action, step_cost in problem.getSuccessors(state):
                if successor not in visited:
                    new_cost = total_cost + step_cost
            #The algorithm sorts and selects the node with the lowest total_cost for expansion.
                    priority_queue.append((new_cost, (successor, actions + [action])))

    return []
util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    priority_queue = [] 
     # list (priority_queue) and sort it based on priority which is calculated as new_cost + heuristic(successor).
    visited = set()
    start_state = problem.getStartState()
    priority_queue.append((heuristic(start_state, problem), (start_state, [])))  # (priority, (state, path))
#A* search expands nodes based on the sum of the path cost (new_cost) and the estimated cost to the goal (heuristic).
    while priority_queue:
        priority_queue.sort(key=lambda x: x[0])  # Sort by priority (cost + heuristic)
        _, (state, actions) = priority_queue.pop(0)

        if problem.isGoalState(state):
            return actions

        if state not in visited:
            visited.add(state)
            for successor, action, step_cost in problem.getSuccessors(state):
                if successor not in visited:

                    new_cost = problem.getCostOfActions(actions + [action])
                #The algorithm selects and expands the node with the lowest priority, ensuring optimal and efficient pathfinding.
                    priority_queue.append((new_cost + heuristic(successor, problem), (successor, actions + [action])))

    return []
util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
