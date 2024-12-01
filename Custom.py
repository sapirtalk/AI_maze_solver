import time

class CustomSolver:
    def __init__(self, maze, start, end, draw_callback):
        self.maze = maze
        self.start = start
        self.end = end
        self.size = len(maze)
        self.last_bound = 0
        self.directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # Possible movements: right, down, left, up
        self.explored_nodes = []  # List to keep track of explored nodes
        self.draw_callback = draw_callback
        self.deadends = set()  # Set to store known dead-end nodes

    def manhattan_distance(self, current, goal):
        # Add a penalty for dead-end nodes
        deadend_penalty = 0
        if self.maze[current[0]][current[1]] > 1:  # Dead-end marker in the grid
            deadend_penalty = self.maze[current[0]][current[1]] - 1  # Higher penalty for higher markers
        return abs(current[0] - goal[0]) + abs(current[1] - goal[1]) + deadend_penalty

    def solve(self):
        bound = self.manhattan_distance(self.start, self.end)
        path = [self.start]

        while True:
            t = self._search(path, 0, bound)
            if t == "FOUND":
                return self.explored_nodes
            if t == float('inf'):
                return None
            bound = t

            if self.draw_callback is not None:
                time.sleep(0.025 / self.size)

    def _search(self, path, g, bound):
        current = path[-1]
        f = g + self.manhattan_distance(current, self.end)

        if f > bound:
            return f
        if current == self.end:
            return "FOUND"

        min_bound = float('inf')
        neighbors = []

        # Collect valid neighbors
        for direction in self.directions:
            neighbor = (current[0] + direction[0], current[1] + direction[1])
            if 0 <= neighbor[0] < self.size and 0 <= neighbor[1] < self.size and self.maze[neighbor[0]][neighbor[1]] == 0:
                if neighbor not in path and neighbor not in self.deadends:
                    neighbors.append(neighbor)

        # If only one neighbor exists, we're in a corridor
        if len(neighbors) == 1:
            corridor = [current]
            while len(neighbors) == 1:
                next_node = neighbors[0]
                corridor.append(next_node)
                neighbors = []
                for direction in self.directions:
                    neighbor = (next_node[0] + direction[0], next_node[1] + direction[1])
                    if (
                        0 <= neighbor[0] < self.size
                        and 0 <= neighbor[1] < self.size
                        and self.maze[neighbor[0]][neighbor[1]] == 0
                        and neighbor not in corridor
                    ):
                        neighbors.append(neighbor)

            # If corridor ends in a dead end, mark it
            if not neighbors:  # Dead end detected
                for node in corridor:
                    self.deadends.add(node)
                    self.maze[node[0]][node[1]] += 1  # Mark corridor in maze grid
                return float('inf')

        # Explore neighbors
        for neighbor in neighbors:
            path.append(neighbor)

            if self.draw_callback is not None:
                self.draw_callback(path)

            t = self._search(path, g + 1, bound)

            if t == "FOUND":
                self.last_bound = bound
                self.explored_nodes.append(neighbor)
                return "FOUND"
            if t < min_bound:
                min_bound = t
            path.pop()

        return min_bound
