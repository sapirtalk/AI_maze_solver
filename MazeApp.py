import tkinter as tk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from Manhatten import ManhattanSolver
from Bi_directional import BiDirectionalManhattanSolver
import random
import time
import tracemalloc


class MazeApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Maze Generator and Solver")
        self.size = 21  
        self.maze = [[1 for _ in range(self.size)] for _ in range(self.size)]  
        self.start = (0, 0)
        self.end = (self.size - 1, self.size - 1)
        self.canvas = None  
        self.create_widgets()

    def create_widgets(self):
        
        self.canvas_frame = tk.Frame(self.root)
        self.canvas_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.control_frame = tk.Frame(self.root)
        self.control_frame.pack(side=tk.BOTTOM)

       
        tk.Label(self.control_frame, text="Maze Size:").pack(side=tk.LEFT)
        self.size_entry = tk.Entry(self.control_frame, width=5)
        self.size_entry.pack(side=tk.LEFT)
        self.size_entry.insert(0, str(self.size))  

        self.generate_button = tk.Button(self.control_frame, text="Generate Maze", command=self.update_maze_size)
        self.generate_button.pack(side=tk.LEFT)

        self.solve_button = tk.Button(self.control_frame, text="Solve Maze", command=lambda: self.solve_maze(self.heuristic_var.get()))
        self.solve_button.pack(side=tk.LEFT)

        self.graph_button = tk.Button(self.control_frame, text="Graph Size/Accuracy", command=lambda: self.graph_size_accuracy(self.heuristic_var.get()))
        self.graph_button.pack(side=tk.LEFT)

        self.solver_compare_button = tk.Button(self.control_frame, text="Compare Solvers", command=self.plot_comparison_chart)
        self.solver_compare_button.pack(side=tk.LEFT)


        self.heuristic_var = tk.StringVar(value="A* with Manhattan heuristic")
        self.heuristic_menu = tk.OptionMenu(self.control_frame, self.heuristic_var, "A* with Manhattan heuristic"  , "Bi-directional A* with Manhattan heuristic")
        self.heuristic_menu.pack(side=tk.LEFT)





    # take the statistics from the specific solver
    def solver_statistics(self, heuristic_var):


        sizes = [100 , 105 , 110 , 115 , 120 , 125 , 130 , 135 , 140 , 145 , 150 , 155 , 160 , 165 , 170 , 175 , 180 , 185 , 190 , 195 , 200]
        accuracies = []
        memory_usages = []
        times_taken = []

        for size in sizes:
            self.size = size
            self.generate_maze()

            if heuristic_var == "A* with Manhattan heuristic":
                solver = ManhattanSolver(self.maze, self.start, self.end)
            elif heuristic_var == "Bi-directional A* with Manhattan heuristic":
                solver = BiDirectionalManhattanSolver(self.maze, self.start, self.end)
            # elif heuristic_var == "IDA* with Manhattan heuristic":
            #     solver = IDAStarSolver(self.maze, self.start, self.end)
            else:
                raise ValueError("Invalid heuristic option")

            tracemalloc.start()  # Start tracking memory
            start_time = time.time()  # Start timing

            path = solver.solve()

            end_time = time.time()  # End timing
            current, peak = tracemalloc.get_traced_memory()  # Get memory usage
            tracemalloc.stop()  # Stop tracking memory

            time_taken = end_time - start_time
            accuracy = len(path) / len(solver.explored_nodes) if solver.explored_nodes else 0

            accuracies.append(accuracy)
            memory_usages.append(peak / (1024 * 1024))  # Convert bytes to megabytes
            times_taken.append(time_taken)

        return accuracies, memory_usages, times_taken

    # calculate the average of the solver statistics
    def compute_averages(self, heuristic_var):
        accuracies, memory_usages, times_taken = self.solver_statistics(heuristic_var)

        avg_accuracy = sum(accuracies) / len(accuracies)
        avg_memory = sum(memory_usages) / len(memory_usages)
        avg_time = sum(times_taken) / len(times_taken)

        return avg_accuracy, avg_memory, avg_time
    

    def plot_comparison_chart(self):
        solvers = [
            "A* with Manhattan heuristic",
            "Bi-directional A* with Manhattan heuristic"
        ]

        avg_accuracies = []
        avg_memory_usages = []
        avg_times_taken = []

        for solver in solvers:
            avg_accuracy, avg_memory, avg_time = self.compute_averages(solver)
            avg_accuracies.append(avg_accuracy)
            avg_memory_usages.append(avg_memory)
            avg_times_taken.append(avg_time)

        bar_width = 0.3
        indices = range(len(solvers))

        plt.figure(figsize=(10, 6))

        # Plotting the bars for memory usage and time taken
        memory_bars = plt.bar([i - bar_width/2 for i in indices], avg_memory_usages, width=bar_width, label='Memory Usage (MB)')
        time_bars = plt.bar([i + bar_width/2 for i in indices], avg_times_taken, width=bar_width, label='Time Taken (s)')

       # Annotating the accuracy as a number in the middle of the bars
        for i, accuracy in enumerate(avg_accuracies):
            # Calculate the midpoint between the two bars
            midpoint = (avg_memory_usages[i] + avg_times_taken[i]) / 2
            plt.text(i, midpoint, f'Accuracy: {accuracy:.2%}', 
                    ha='center', va='center', fontsize=12, color='black')


        plt.xticks(indices, solvers)
        plt.xlabel('Solvers')
        plt.ylabel('Values (AVG)')
        plt.title('Comparison of Solvers Approach (Large Search Space Mazes)')
        plt.legend()

        plt.show()






    def graph_size_accuracy(self , heuristic_var):
        print("Graphing size and accuracy")

        sizes = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
        accuracies = []
        for size in sizes:
            self.size = size
            self.generate_maze()

            if heuristic_var == "A* with Manhattan heuristic":
                solver = ManhattanSolver(self.maze, self.start, self.end)
            elif heuristic_var == "Bi-directional A* with Manhattan heuristic":
                solver = BiDirectionalManhattanSolver(self.maze, self.start, self.end)
            else:
                raise ValueError("Invalid heuristic option")

            path = solver.solve()
            accuracy = len(path) / len(solver.explored_nodes) if solver.explored_nodes else 0
            accuracies.append(accuracy)


        # make the line graph smoother
        for i in range(1, len(accuracies) - 1):
            accuracies[i] = (accuracies[i - 1] + accuracies[i] + accuracies[i + 1]) / 3

        plt.plot(sizes, accuracies)
        
        plt.xlabel("Maze Size")
        plt.ylabel("Accuracy")
        plt.title("Accuracy vs. Maze Size - " + heuristic_var)
        plt.show()
        
    
    
    
    
    
    def update_maze_size(self):
        try:
            size = int(self.size_entry.get())
            if size < 5:
                raise ValueError("Size must be greater than 4")
            if size % 2 == 0:
                size += 1  # Ensure size is odd , better for path finding
            self.size = size
            self.generate_maze()
        except ValueError:
            print("Invalid size input. Please enter a positive integer greater than 4.")

    def generate_maze(self):
        self.maze = [[1 for _ in range(self.size)] for _ in range(self.size)]  # Reset maze with walls
        self.start = (0, 0)
        self.end = (self.size - 1, self.size - 1)
        self.prim_maze_generation()
        self.ensure_clear_start_end()
        self.draw_maze()

    def prim_maze_generation(self):
        start_x, start_y = 1, 1
        self.maze[start_y][start_x] = 0
        walls = [(start_x + dx, start_y + dy) for dx, dy in [(0, 2), (2, 0), (0, -2), (-2, 0)]]
        random.shuffle(walls)

        while walls:
            wall = walls.pop()
            x, y = wall

            if 0 <= x < self.size and 0 <= y < self.size and self.maze[y][x] == 1:
                adjacent_cells = []
                if x >= 2 and self.maze[y][x - 2] == 0:
                    adjacent_cells.append((x - 2, y))
                if x < self.size - 2 and self.maze[y][x + 2] == 0:
                    adjacent_cells.append((x + 2, y))
                if y >= 2 and self.maze[y - 2][x] == 0:
                    adjacent_cells.append((x, y - 2))
                if y < self.size - 2 and self.maze[y + 2][x] == 0:
                    adjacent_cells.append((x, y + 2))

                if adjacent_cells:
                    new_x, new_y = random.choice(adjacent_cells)
                    self.maze[y][x] = 0
                    self.maze[(y + new_y) // 2][(x + new_x) // 2] = 0  # Break the wall
                    self.maze[new_y][new_x] = 0

                    walls.extend([(x + dx, y + dy) for dx, dy in [(0, 2), (2, 0), (0, -2), (-2, 0)] if
                                  0 <= x + dx < self.size and 0 <= y + dy < self.size and self.maze[y + dy][x + dx] == 1])

                    random.shuffle(walls)

    def ensure_clear_start_end(self):
        # Clear paths around the start point
        start_x, start_y = self.start
        self.maze[start_y][start_x] = 0
        if start_y + 1 < self.size:
            self.maze[start_y + 1][start_x] = 0
        # if start_x + 1 < self.size:
        #     self.maze[start_y][start_x + 1] = 0
        
        # Clear paths around the end point
        end_x, end_y = self.end
        self.maze[end_y][end_x] = 0
        if end_y - 1 >= 0:
            self.maze[end_y - 1][end_x] = 0
        # if end_x - 1 >= 0:
        #     self.maze[end_y][end_x - 1] = 0

    def draw_maze(self):
        if self.canvas:
            self.canvas.get_tk_widget().destroy()  # Clear the previous canvas if it exists

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(self.maze, cmap='binary')
        ax.scatter(self.start[1], self.start[0], c='green', s=100)  # Start point
        ax.scatter(self.end[1], self.end[0], c='red', s=100)  # End point
        ax.set_xticks([])
        ax.set_yticks([])
        plt.close(fig)  # Close the figure to prevent it from displaying in a separate window

        # Embed the figure in the Tkinter canvas
        self.canvas = FigureCanvasTkAgg(fig, master=self.canvas_frame)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.canvas.draw()

    def solve_maze(self , heuristic_var = "A* with Manhattan heuristic"):
        

        if heuristic_var == "A* with Manhattan heuristic":
            solver = ManhattanSolver(self.maze , self.start , self.end)
        elif heuristic_var == "Bi-directional A* with Manhattan heuristic":
            solver = BiDirectionalManhattanSolver(self.maze , self.start , self.end)
        else :
            raise ValueError("invalid heuristic")



        path = solver.solve()

        if path:
            accuracy = len(path) / len(solver.explored_nodes) if solver.explored_nodes else 0
            self.draw_explored_path(solver.explored_nodes , path , accuracy)  # Draw the explored nodes after the solution
        else:
            print("No path found.")
            self.draw_explored_path(solver.explored_nodes)  # Still draw the explored nodes if no solution

    def draw_explored_path(self, explored_nodes , path = None , accuracy = 0):
        if self.canvas:
            self.canvas.get_tk_widget().destroy()  # Clear the previous canvas if it exists

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(self.maze, cmap='binary')

        # Draw the explored nodes
        for node in explored_nodes:

            if path and node not in path:
                ax.scatter(node[1], node[0], c='yellow', s=50) 
            else :
                ax.scatter(node[1], node[0], c='blue', s=50)    

        ax.scatter(self.start[1], self.start[0], c='green', s=50)  # Start point
        ax.scatter(self.end[1], self.end[0], c='red', s=50)  # End point

        # Display the accuracy on the plot
        ax.text(0.5, -0.1, f"Accuracy: {accuracy:.2%}", ha='center', va='center', transform=ax.transAxes, fontsize=12, color='black')


        ax.set_xticks([])
        ax.set_yticks([])
        plt.close(fig)  # Close the figure to prevent it from displaying in a separate window

        # Embed the figure in the Tkinter canvas
        self.canvas = FigureCanvasTkAgg(fig, master=self.canvas_frame)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.canvas.draw()

if __name__ == "__main__":
    root = tk.Tk()
    app = MazeApp(root)
    root.mainloop()