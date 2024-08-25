import time

class CustomSolver:
    def __init__(self, maze, start, end , draw_callback):
        self.maze = maze
        self.start = start
        self.end = end
        self.size = len(maze)
        self.directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # Possible movements: right, down, left, up
        self.explored_nodes = []  # List to keep track of explored nodes
        self.draw_callback = draw_callback


    def manhattan_distance(self, current, goal):
        return abs(current[0] - goal[0]) + abs(current[1] - goal[1])
    

    def solve(self):
        pass

            

        



        

