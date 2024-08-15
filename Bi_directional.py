import heapq
import time

class BiDirectionalManhattanSolver:
    def __init__(self, maze, start, end , draw_callback):
        self.maze = maze
        self.start = start
        self.end = end
        self.size = len(maze)
        self.directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # Possible movements: right, down, left, up
        self.explored_nodes_forward = []  # List to keep track of explored nodes in the forward search
        self.explored_nodes_backward = []  # List to keep track of explored nodes in the backward search
        self.explored_nodes = []  # List to keep track of explored nodes in both searches
        self.draw_callback = draw_callback

    def manhattan_distance(self, current, goal):
        return abs(current[0] - goal[0]) + abs(current[1] - goal[1])

    def solve(self):
        # Initialize the open sets for forward and backward search
        open_set_forward = []
        open_set_backward = []
        heapq.heappush(open_set_forward, (0, self.start))
        heapq.heappush(open_set_backward, (0, self.end))

        # Initialize the g_score, f_score and came_from dictionaries for both searches
        came_from_forward = {}
        came_from_backward = {}
        g_score_forward = {self.start: 0}
        g_score_backward = {self.end: 0}
        f_score_forward = {self.start: self.manhattan_distance(self.start, self.end)}
        f_score_backward = {self.end: self.manhattan_distance(self.end, self.start)}

        # Sets to keep track of visited nodes
        visited_forward = set()
        visited_backward = set()

        while open_set_forward and open_set_backward:
            # Expand the forward search
            current_forward = self.expand_search(open_set_forward, visited_forward, g_score_forward, f_score_forward, came_from_forward, self.end)
            self.explored_nodes_forward.append(current_forward)
            self.explored_nodes.append(current_forward)

            # Expand the backward search
            current_backward = self.expand_search(open_set_backward, visited_backward, g_score_backward, f_score_backward, came_from_backward, self.start)
            self.explored_nodes_backward.append(current_backward)
            self.explored_nodes.append(current_backward)

            if self.draw_callback is not None:
                self.draw_callback(self.explored_nodes)

            # Check if the searches meet
            if current_forward in visited_backward:
                return self.reconstruct_path(came_from_forward, came_from_backward, current_forward)
            if current_backward in visited_forward:
                return self.reconstruct_path(came_from_backward, came_from_forward, current_backward, reverse=True)

            if self.draw_callback is not None:
                time.sleep(0.1)
        return None  # No path found

    def expand_search(self, open_set, visited, g_score, f_score, came_from, goal):
        _, current = heapq.heappop(open_set)
        visited.add(current)

        for direction in self.directions:
            neighbor = (current[0] + direction[0], current[1] + direction[1])

            if 0 <= neighbor[0] < self.size and 0 <= neighbor[1] < self.size and self.maze[neighbor[0]][neighbor[1]] == 0:
                tentative_g_score = g_score[current] + 1

                if tentative_g_score < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + self.manhattan_distance(neighbor, goal)
                    if neighbor not in visited:
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))

        return current

    def reconstruct_path(self, came_from_forward, came_from_backward, meeting_point, reverse=False):
        # Reconstruct the path from start to meeting point
        path_forward = [meeting_point]
        current = meeting_point
        while current in came_from_forward:
            current = came_from_forward[current]
            path_forward.append(current)
        path_forward.reverse()

        # Reconstruct the path from end to meeting point
        path_backward = []
        current = meeting_point
        while current in came_from_backward:
            current = came_from_backward[current]
            path_backward.append(current)

        # Combine both paths
        if reverse:
            path_forward, path_backward = path_backward, path_forward

        return path_forward + path_backward[1:]  # Exclude the meeting point from the backward path 
