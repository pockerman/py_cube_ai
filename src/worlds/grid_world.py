"""
GridWorld environment. The environment is taken from the
book Deep Reinforcement Learning in Action and adapted
to fit the environment API.

The Gridworld board is always square. There are four objects in the world

- Player
- Goal
- Pit
-Wall

The state is represented as a side_size × side_size × side_size tensor.
The first dimension indexes a set of four matrices of size side_size × side_size.
You can interpret this as having the dimensions frames by height by width.
Each matrix is a side_size × side_size grid of zeros and a single 1, where a 1 indicates the position of a particular object.
Each matrix encodes the position of one of the four objects: the player, the goal, the pit, and the wall.

- The first matrix encodes the position of the player,
- The second matrix encodes the position of the goal,
- The third matrix encodes the position of the pit,
- The fourth matrix encodes the position of the wall.

"""
import enum
import numpy as np
from src.worlds.grid_board import *
from src.utils.time_step import TimeStep, StepType


class GridWorldActionType(enum.IntEnum):
    """
    Enumeration of actions for GridWorld
    """
    INVALID = -1
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3


class GridworldInitMode(enum.IntEnum):
    """
    Enumeration of initialization mode
    """
    INVALID = -1
    STATIC = 0
    PLAYER = 1
    RANDOM = 2


class Gridworld(object):



    """
    The Gridworld class models a square board. There are three ways to initialize the board. 
    - static
    - random
    - player
    See the InitMode class.
    Static initialization means that the objects on the board are initialized at the same predetermined locations. 
    Player initialization means that the player is initialized at a random position on the board. 
    Random initialization means that all the objects are placed randomly  
    """

    VALID_ACTIONS = [GridWorldActionType.UP, GridWorldActionType.DOWN,
                     GridWorldActionType.LEFT, GridWorldActionType.RIGHT]

    VALID_INIT_MODES = [GridworldInitMode.STATIC, GridworldInitMode.PLAYER, GridworldInitMode.RANDOM]

    @staticmethod
    def get_action(aidx: int) -> GridWorldActionType:
        """
        Returns the action that corresponds to the
        given action index
        :param aidx:
        :return:
        """
        return Gridworld.VALID_ACTIONS[aidx]

    def __init__(self, size: int = 4, mode: GridworldInitMode = GridworldInitMode.STATIC,
                 add_noise_on_state: bool = True) -> None:
        """
        The Gridworld board is always square, so the size refers to one side’s dimension 4 × 4 grid will be created.
        :param size: The size of the board in every dimension
        :param mode: Initialization mode
        """

        if size < 4:
            raise ValueError("Minimum board size is 4. Requested board size of {0}".format(size))

        if mode not in Gridworld.VALID_INIT_MODES:
            raise ValueError("Mode {0} not in {1}".format(mode, Gridworld.VALID_INIT_MODES))

        self.size = size
        self.mode = mode
        self.add_noise_on_state = add_noise_on_state
        self.board: GridBoard = None
        self._initialize()

    @property
    def n_actions(self) -> int:
        return len(Gridworld.VALID_ACTIONS)

    def reset(self) -> TimeStep:
        """
        Reset the environment
        :return:
        """
        self._initialize()

        if self.add_noise_on_state:
            obs = self.board.render_np().reshape(1, 64) + np.random.rand(1, 64) / 10.0
        else:
            obs = self.board.render_np().reshape(1, 64)

        time_step = TimeStep(step_type=StepType.FIRST, reward=self.get_reward(),
                             info={}, observation=obs, discount=0.0)
        return time_step

    def step(self, action: GridWorldActionType) -> TimeStep:
        """
        Step into the environment
        :param action:
        :return:
        """

        if action not in Gridworld.VALID_ACTIONS:
            raise ValueError("{0} not in {1}".format(action.name, Gridworld.VALID_ACTIONS))

        # need to determine what object (if any)
        # is in the new grid spot the player is moving to
        # actions in {u,d,l,r}
        def checkMove(addpos):
            if self.validateMove('Player', addpos) in [0, 2]:
                new_pos = addTuple(self.board.components['Player'].pos, addpos)
                self.board.movePiece('Player', new_pos)

        if action == GridWorldActionType.UP: #'u': #up
            checkMove((-1, 0))
        elif action == GridWorldActionType.DOWN: #'d': #down
            checkMove((1, 0))
        elif action == GridWorldActionType.LEFT: #'l': #left
            checkMove((0, -1))
        elif action == GridWorldActionType.RIGHT: #'r': #right
            checkMove((0, 1))

        if self.add_noise_on_state:
            obs = self.board.render_np().reshape(1, 64) + np.random.rand(1, 64) / 10.0
        else:
            obs = self.board.render_np().reshape(1, 64)

        reward = self.get_reward()
        step_type = StepType.LAST if reward != -1 else StepType.MID
        time_step = TimeStep(step_type=step_type, reward=reward,
                             info={}, observation=obs, discount=0.0)
        return time_step

    def _initialize(self) -> None:
        """
        Initialize the world
        :return:
        """

        # set up the board
        self.board = GridBoard(size=self.size)

        # Add pieces, positions will be updated later
        self.board.addPiece('Player', 'P', (0, 0))
        self.board.addPiece('Goal', '+', (1, 0))
        self.board.addPiece('Pit', '-', (2, 0))
        self.board.addPiece('Wall', 'W', (3, 0))

        if self.mode == GridworldInitMode.STATIC: #'static':
            self.initGridStatic()
        elif self.mode == GridworldInitMode.PLAYER: #'player':
            self.initGridPlayer()
        else:
            self.initGridRand()


    #Initialize stationary grid, all items are placed deterministically
    def initGridStatic(self):
        #Setup static pieces
        self.board.components['Player'].pos = (0,3) #Row, Column
        self.board.components['Goal'].pos = (0,0)
        self.board.components['Pit'].pos = (0,1)
        self.board.components['Wall'].pos = (1,1)

    #Check if board is initialized appropriately (no overlapping pieces)
    #also remove impossible-to-win boards
    def validateBoard(self):
        valid = True

        player = self.board.components['Player']
        goal = self.board.components['Goal']
        wall = self.board.components['Wall']
        pit = self.board.components['Pit']

        all_positions = [piece for name,piece in self.board.components.items()]
        all_positions = [player.pos, goal.pos, wall.pos, pit.pos]
        if len(all_positions) > len(set(all_positions)):
            return False

        corners = [(0,0),(0,self.board.size), (self.board.size,0), (self.board.size,self.board.size)]
        #if player is in corner, can it move? if goal is in corner, is it blocked?
        if player.pos in corners or goal.pos in corners:
            val_move_pl = [self.validateMove('Player', addpos) for addpos in [(0,1),(1,0),(-1,0),(0,-1)]]
            val_move_go = [self.validateMove('Goal', addpos) for addpos in [(0,1),(1,0),(-1,0),(0,-1)]]
            if 0 not in val_move_pl or 0 not in val_move_go:
                #print(self.display())
                #print("Invalid board. Re-initializing...")
                valid = False

        return valid

    #Initialize player in random location, but keep wall, goal and pit stationary
    def initGridPlayer(self):
        #height x width x depth (number of pieces)
        self.initGridStatic()
        #place player
        self.board.components['Player'].pos = randPair(0,self.board.size)

        if (not self.validateBoard()):
            #print('Invalid grid. Rebuilding..')
            self.initGridPlayer()

    #Initialize grid so that goal, pit, wall, player are all randomly placed
    def initGridRand(self):
        #height x width x depth (number of pieces)
        self.board.components['Player'].pos = randPair(0,self.board.size)
        self.board.components['Goal'].pos = randPair(0,self.board.size)
        self.board.components['Pit'].pos = randPair(0,self.board.size)
        self.board.components['Wall'].pos = randPair(0,self.board.size)

        if (not self.validateBoard()):
            #print('Invalid grid. Rebuilding..')
            self.initGridRand()

    def validateMove(self, piece, addpos=(0,0)):
        outcome = 0 #0 is valid, 1 invalid, 2 lost game
        pit = self.board.components['Pit'].pos
        wall = self.board.components['Wall'].pos
        new_pos = addTuple(self.board.components[piece].pos, addpos)
        if new_pos == wall:
            outcome = 1 #block move, player can't move to wall
        elif max(new_pos) > (self.board.size-1):    #if outside bounds of board
            outcome = 1
        elif min(new_pos) < 0: #if outside bounds
            outcome = 1
        elif new_pos == pit:
            outcome = 2

        return outcome

    def get_reward(self) -> int:
        if (self.board.components['Player'].pos == self.board.components['Pit'].pos):
            return -10
        elif (self.board.components['Player'].pos == self.board.components['Goal'].pos):
            return 10
        else:
            return -1

    def render(self) -> np.array:
        """
        Renders the board on the standard output
        :return:
        """
        board = self.board.render()
        print(board)
        return board
