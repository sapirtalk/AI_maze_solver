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
        self.dead_ends = set()  # Use a set for efficient lookups
        self.cur_corridor = []  # Tracks the current corridor nodes

    def manhattan_distance(self, current, goal):
        return abs(current[0] - goal[0]) + abs(current[1] - goal[1])

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

        # Find valid neighbors
        neighbors = []
        for direction in self.directions:
            neighbor = (current[0] + direction[0], current[1] + direction[1])
            if (
                0 <= neighbor[0] < self.size
                and 0 <= neighbor[1] < self.size
                and self.maze[neighbor[0]][neighbor[1]] == 0  # Open path
                and neighbor not in self.dead_ends  # Not a dead end
            ):
                neighbors.append(neighbor)

        # Sort neighbors by Manhattan distance
        print(f"Current node: {current}, Neighbors: {neighbors}")
        neighbors.sort(key=lambda n: self.manhattan_distance(n, self.end))

        # Detect corridor (exactly one way forward)
        if len(neighbors) < 3:
            print ("Corridor detected:", neighbors[0])
            if not self.cur_corridor:  # Start a new corridor
                self.cur_corridor.append(current)
                print ("New corridor started:", current)
            self.cur_corridor.append(neighbors[0])

        # Dead end detection
        if len(neighbors) == 1:
            print ("Dead end detected:", current)
            if self.cur_corridor:
                # Mark only the first node in the corridor as a dead end
                first_corridor_node = self.cur_corridor[0]
                self.dead_ends.add(first_corridor_node)
                print ("Dead end found:", first_corridor_node)
            self.cur_corridor.clear()  # Clear the corridor
            return float('inf')

        # Explore neighbors
        for neighbor in neighbors:
            if neighbor not in path:
                path.append(neighbor)

                # Clear the corridor if branching occurs
                if len(neighbors) > 1:
                    self.cur_corridor.clear()

                if self.draw_callback is not None:
                    self.draw_callback(path, dead_ends=self.dead_ends)

                t = self._search(path, g + 1, bound)

                if t == "FOUND":
                    self.last_bound = bound
                    self.explored_nodes.append(neighbor)
                    return "FOUND"
                if t < min_bound:
                    min_bound = t

                path.pop()

                # Remove from corridor if backtracking
                if neighbor in self.cur_corridor:
                    self.cur_corridor.remove(neighbor)

        return min_bound
