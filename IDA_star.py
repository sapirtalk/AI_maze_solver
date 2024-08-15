import time

class IDAStarSolver:
    def __init__(self, maze, start, end , draw_callback):
        self.maze = maze
        self.start = start
        self.end = end
        self.size = len(maze)
        self.last_bound = 0
        self.directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # Possible movements: right, down, left, up
        self.explored_nodes = []  # List to keep track of explored nodes
        self.draw_callback = draw_callback

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
                time.sleep(0.025/self.size)
        
            

    def _search(self, path, g, bound):
        current = path[-1]
        f = g + self.manhattan_distance(current, self.end)
        
        if f > bound:
            return f
        if current == self.end:
            return "FOUND"

        min_bound = float('inf')

        for direction in self.directions:
            neighbor = (current[0] + direction[0], current[1] + direction[1])

            if 0 <= neighbor[0] < self.size and 0 <= neighbor[1] < self.size and self.maze[neighbor[0]][neighbor[1]] == 0:
                if neighbor not in path:  # Avoid cycles
                    path.append(neighbor)
                    
                    if self.draw_callback is not None:
                        self.draw_callback(path)
                    
                    t = self._search(path, g + 1, bound)

                    if t == "FOUND":
                        self.last_bound = bound
                        self.explored_nodes.append((self.end[0]-1, self.end[1]))
                        return "FOUND"
                    if t < min_bound:
                        min_bound = t
                    path.pop()

        return min_bound
