import heapq


class ManhattanSolver:
    def __init__(self, maze, start, end):
        self.maze = maze
        self.start = start
        self.end = end
        self.size = len(maze)
        self.directions = [(0, 1), (1, 0), (0, -1), (-1, 0)] 
        self.explored_nodes = []  

    def manhattan_distance(self, current, goal):
        return abs(current[0] - goal[0]) + abs(current[1] - goal[1])

    def solve(self):
        open_set = []
        heapq.heappush(open_set, (0, self.start))
        came_from = {}
        g_score = {self.start: 0}
        f_score = {self.start: self.manhattan_distance(self.start, self.end)}

        while open_set:
            _, current = heapq.heappop(open_set)

            self.explored_nodes.append(current) 

            if current == self.end:
                return self.reconstruct_path(came_from, current)

            for direction in self.directions:
                neighbor = (current[0] + direction[0], current[1] + direction[1])

                if 0 <= neighbor[0] < self.size and 0 <= neighbor[1] < self.size and self.maze[neighbor[0]][neighbor[1]] == 0:
                    tentative_g_score = g_score[current] + 1

                    if tentative_g_score < g_score.get(neighbor, float('inf')):
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g_score
                        f_score[neighbor] = tentative_g_score + self.manhattan_distance(neighbor, self.end)
                        if neighbor not in [i[1] for i in open_set]:
                            heapq.heappush(open_set, (f_score[neighbor], neighbor))

        return None 

    def reconstruct_path(self, came_from, current):
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path
