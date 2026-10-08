import math


def task_weight(task):
    """Assigned task weight; legacy four-field tasks weigh one unit."""
    weight = task[4] if len(task) > 4 else 1.0
    if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not math.isfinite(weight) or weight <= 0:
        raise ValueError("Task weight must be finite and positive")
    return float(weight)


class Agent:
    def __init__(self, agent_id : int, state : tuple, task_sequence : list = None, home : tuple = None, capacity: int = 1):
        if isinstance(capacity, bool) or not isinstance(capacity, (int, float)) or not math.isfinite(capacity) or capacity <= 0:
            raise ValueError("Agent capacity must be finite and positive")
        self.capacity = capacity
        self.carried_items = {}  # task_id -> SKU; duplicate SKUs are distinct items
        self.schedule = None  # CBTA events: (task_id, phase), 1 pickup / 2 delivery
        self.id = agent_id
        self.task_sequence = task_sequence if task_sequence is not None else []
        self.path_sequence = []
        self.state = state
        if home is None:
            self.home = (state[0], state[1])
        else:
            self.home = home
        # 0 == Free Agent
        # 1 == To_Pickup
        # 2 == To_Delivery
        self.status = 0
        self.sku_id_carrying = None
    
    def set_sku_id_carrying(self, sku_id : int):
        if self.schedule is not None:
            task_id = self.task_sequence[0][0]
            if sku_id is None:
                self.carried_items.pop(task_id)
            else:
                if task_id in self.carried_items or self.carried_weight + task_weight(self.task_sequence[0]) > self.capacity + 1e-9:
                    raise ValueError("Pickup would exceed capacity or duplicate an item")
                self.carried_items[task_id] = sku_id
            return
        self.sku_id_carrying = sku_id
    
    def get_sku_id_carrying(self) -> int:
        if self.schedule is not None:
            return self.carried_items.get(self.task_sequence[0][0]) if self.task_sequence else None
        return self.sku_id_carrying

    @property
    def assigned_weight(self):
        return sum(task_weight(task) for task in self.task_sequence)

    @property
    def carried_weight(self):
        return sum(task_weight(task) for task in self.task_sequence if task[0] in self.carried_items)

    @property
    def remaining_capacity(self):
        return max(0.0, self.capacity - self.assigned_weight)

    def enable_schedule(self):
        if self.schedule is not None:
            return
        if self.assigned_weight > self.capacity + 1e-9:
            raise ValueError("Assigned task weights exceed robot capacity")
        self.schedule = []
        for i, task in enumerate(self.task_sequence):
            if i == 0 and self.status == 2:
                if self.sku_id_carrying is None:
                    raise ValueError("Delivery task has no carried SKU")
                self.carried_items[task[0]] = self.sku_id_carrying
            else:
                self.schedule.append((task[0], 1))
            self.schedule.append((task[0], 2))
        self.sync_schedule()

    def sync_schedule(self):
        """Expose the next event through the existing status/first-task interface."""
        if not self.schedule:
            self.status = 0
            return
        task_id, self.status = self.schedule[0]
        index = next(i for i, task in enumerate(self.task_sequence) if task[0] == task_id)
        self.task_sequence.insert(0, self.task_sequence.pop(index))

    def set_active_on_task(self, status : int):
        """Sets Status of the Agent
        0 == Free Agent
        1 == To_Pickup
        2 == To_Delivery

        Args:
            status (int): Current status of agent
        """
        self.active_on_task = status
    
    def set_agent_id(self, agent_id):
        self.id = agent_id

    def get_agent_id(self):
        return self.id

    def set_task_sequence(self, task_sequences : list):
        self.task_sequences = task_sequences
    
    def get_assigned_task_ids(self) -> list:
        return [task_id for task_id in self.task_sequence]
    
    def set_path_sequences(self, path_sequences : list):
        self.path_sequence = path_sequences
        
    def add_to_path_sequences(self, path_sequence):
        self.path_sequence.append(path_sequence)
    
    def __str__(self):
        return f"Agent ID: {self.id}, State: {self.state}"
    
    
class AgentLoader:
    def __init__(self, agents : list):
        self.agents = agents
        
    def get_agent(self, agent_id) -> Agent:
        for agent in self.agents:
            if agent.id == agent_id:
                return agent
        
    def get_agent_states(self) -> list:
        return [agent.state for agent in self.agents]

    def get_free_agents(self):
        return [agent for agent in self.agents if agent.status == 0]
    
    def get_to_pickup_agents(self):
        return [agent for agent in self.agents if agent.status == 1] 
    
    def get_to_delivery_agents(self):
        return [agent for agent in self.agents if agent.status == 2]
    
    def get_all_assigned_tasks_per_robot(self):
        return [agent.get_assigned_task_ids() for agent in self.agents]   
    
    def get_all_agent_carrying_skus(self) -> list:
        skus = []
        for agent in self.agents:
            skus.append(list(agent.carried_items.values()) if agent.schedule is not None else agent.get_sku_id_carrying())
        
        return skus
    
    def get_all_assigned_tasks(self): 
        tasks = []
        for agent in self.agents:
            for task in agent.task_sequence:
                tasks.append(task)
        return tasks
    
    def get_assigned_agent(self, task_id) -> int:
        for agent in self.agents:
            if task_id in agent.task_sequence:
                return agent.id
    
    def detect_collisions(self) -> int:
        return len(self.agents) - len(set(self.get_agent_states()))

    def __str__(self):
        return f"AgentLoader with {len(self.agents)} agents out of {len(self.agents)} allowed"
    
    def copy(self):
        """Create a deep copy of the current solution."""
        new_solution = AgentLoader([])
        for agent in self.agents:
            new_agent = agent.__class__(agent.id, agent.state, task_sequence=agent.task_sequence.copy(), home=agent.home, capacity=agent.capacity)
            new_agent.carried_items = agent.carried_items.copy()
            new_agent.schedule = agent.schedule.copy() if agent.schedule is not None else None
            new_agent.status = agent.status
            new_agent.path_sequence = agent.path_sequence.copy()
            new_agent.sku_id_carrying = agent.sku_id_carrying
            new_solution.agents.append(new_agent)
        return new_solution
