import hashlib

import carb
import numpy as np
import omni.anim.navigation.core as nav

from .randomizer_util import RandomizerUtil

"""
Base Class for Command
    setup_command() and get_parameters() need to be called before getting the commands
    Each agent has different command and each command should be based on this base class
    Each command should implement its own get_parameters() method
"""


class Command:
    def __init__(self, agent=None, agent_list=None, navmesh=None, agent_speed=1):
        self.name = self.__class__.__name__
        self.inav = navmesh
        self.duration = -1  # estimated duration for the command
        self.parameter = ""  # parameters for the command
        self.agent = agent  # which agent does this command belong to
        self.agent_list = agent_list  # some commands need to querry and update the positions for all agents
        self.agent_speed = agent_speed  # how fast does the agent move
        self.num_precision = (
            2  # How many digits to keep as the command parameter, can be overwritten for a specific command
        )

    # Set up the command, need to be called after a Command subclass instance is created
    def setup_command(self, agent, agent_list, navmesh):
        self.agent = agent
        self.agent_list = agent_list
        self.inav = navmesh
        self.get_parameters()

    # Need to be implemented in the subclass. The most essential method for a randomized command
    def get_parameters(self):
        carb.log_error(self.name + " does not have get_parameters() implemented")

    # For the randomizer, a command is invalid if its parameters are empty
    def is_valid(self):
        if self.parameter == "":
            return False
        return True

    # Duration will be set when get_parameters() is called
    def get_duration(self):
        if self.duration < 0:
            carb.log_error("Command " + self.name + " duration is accessed before set")
            return -1
        return self.duration

    # Get the actual command string that will be stored in the command text file
    def get_command(self):
        return self.agent + " " + self.name + " " + self.parameter


"""
Base Randomizer Class
    Contains the method for generating random position and get the random commands for the given agent
    Each randomizer should contain its customized randomization methods
"""


class Randomizer:
    ID_counter = -1

    def __init__(self, global_seed):
        self._global_seed = global_seed
        self._rand_state = None  # used to restore the position in the random sequence and continue the random sequence
        self._existing_pos = []  # cache the random position already generated
        self.agent_name = self.__class__.__name__.replace("Randomizer", "")  # child class name
        self.agent_positions = (
            []
        )  # store the positions for all the agents in the scene to avoid overlaps among different agents
        # Following variables need to be initialized in the children randomizers
        Randomizer.ID_counter += 1
        self.agent_id = Randomizer.ID_counter  # an offset for differentiating spawn locations for different agents.
        self.command_list = []  # list of available commands for this agent
        self.command_map = {}  # map the command string to the command class
        self.transition_matrix = []  # The transition matrix for the command generation Markov Chain
        self.default_command = (
            None  # in the case of generating an invalid command, this default command will be used to replace it
        )
        # Default AABB min and MAX
        self.extent = [(0, 0), (0, 0)]

    # Create a command with given index
    def create_command(self, command_idx, agent, agent_list, inav):
        command = self.command_map[self.command_list[command_idx]]
        command.setup_command(agent, agent_list, inav)
        # if the command has empty paramter, it is an invalid command and will be reset to the default command
        if not command.is_valid():
            command = self.default_command
            if self.default_command == None:
                carb.log_error("Default command not set for " + self.name)
                return
            command.setup_command(agent, agent_list, inav)
        return command

    # Every time global seed is changed, the randomized state needs to be reset
    def update_seed(self, new_seed):
        self._global_seed = new_seed
        self.reset()

    # Need to be called after another agent called spawn()
    def update_agent_positions(self, pos):
        self.agent_positions = pos

    # Reset the randomization state and position cache
    # Should be called when a new environment is loaded
    def reset(self):
        self._rand_state = None
        self._existing_pos = []

    # Generate random commands for the given agent list
    def generate_commands(self, global_seed, duration, agent_list):
        # Get the navmesh in the stage
        inav = nav.acquire_interface()
        # the list that stores all the command
        commands = []

        # Generate commands for the agents one by one
        # agent is a the name of the agent in stage
        for agent in agent_list:
            t = 0  # Total command duration for this agent

            # This helps each agent generate unique seed
            # Note: Hash collision may still happen, although practically it is very unlikely due to the sheer size of the space
            agent_name_hash = int(hashlib.sha256(agent.encode()).hexdigest(), 16) % 10000
            command_seed = RandomizerUtil.handle_overflow(global_seed + agent_name_hash)

            # First command is generated randomly with a uniform probability
            # Future commands are generated according to the givevn Markov Chain
            np.random.seed(command_seed)
            init_command_idx = np.random.randint(0, len(self.command_list))
            try:
                init_command = self.create_command(init_command_idx, agent, agent_list, inav)
            except KeyError:
                raise ValueError("Command class not implemented")
            t += init_command.get_duration()
            # Get the actual text for the command
            commands.append(init_command.get_command())
            current_command = init_command

            # Keep adding commands until it is guaranteed to run for the given duration
            while t < duration:
                # The new command is generated based on the current command's markov chain trainsition matrix
                new_command_idx = np.random.choice(
                    np.array(range(len(self.command_list))), p=np.array(self.transition_matrix[current_command.name])
                )
                try:
                    new_command = self.create_command(new_command_idx, agent, agent_list, inav)
                except KeyError:
                    raise ValueError("Command class not implemented")
                t += new_command.get_duration()
                commands.append(new_command.get_command())
                current_command = new_command
        return commands

    # Randomly generate a valid agent position
    # Along with the global seed, each idx gives a deterministic result
    def get_random_position(self, idx):
        # Pos has been spawned, no need to re-compute
        if idx < len(self._existing_pos):
            return self._existing_pos[idx]

        # Get the navmesh in the stage
        inav = nav.acquire_interface()

        # Restore the previous random state
        if len(self._existing_pos) > 0:
            np.random.set_state(self._rand_state)
        # First time generating random positions
        else:
            spawn_seed = RandomizerUtil.handle_overflow(self._global_seed + self.agent_id)
            np.random.seed(spawn_seed)

        valid = False  # whether the point is a valid point on the navmesh
        spawn_location = carb.Float3(0, 0, 0)
        num_attempts = 0

        spawn_apothem = RandomizerUtil.get_spawn_apothem()
        while not valid:
            # Agents will be spawned within a square of this given apothem, which can be configured via carb.settings
            x = np.random.uniform(-spawn_apothem, spawn_apothem)
            y = np.random.uniform(-spawn_apothem, spawn_apothem)

            # Check point validity and try to cast it to the closest valid point
            valid = inav.closest_navmesh_point(carb.Float3(x, y, 0.0), spawn_location, carb.Float3(0.1, 0.1, 0.1))

            # If the point is valid, check if the AABB bounding points are still valid in the navmesh
            if valid:
                for point in RandomizerUtil.get_2d_bounding_points(self.extent[0], self.extent[1], spawn_location):
                    if not inav.validate_navmesh_point(point, half_extents=carb.Float3(0.2, 0.2, 0.2)):
                        valid = False
                        break

            # If the point is valid for the AABB points on the plane, check if the position overlaps with other agents
            if valid:
                # Avoid position overlaps between agents
                for pos in self._existing_pos + self.agent_positions:
                    if RandomizerUtil.dist3(carb.Float3(pos), spawn_location) < 1:
                        valid = False
                        break
            num_attempts += 1

            # If after 10000 attempts, agent position overlapping is unavoidable, just keep the current position and give a warning
            if num_attempts > 10000:
                carb.log_warn(
                    "With the current number of agents and the scene asset, agent overlapping may not be avoided"
                )
                break

        # Store the random number state so next time it continues the sequence
        self._existing_pos.append(spawn_location)
        self._rand_state = np.random.get_state()
        return spawn_location
