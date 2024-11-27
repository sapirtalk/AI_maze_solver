import tkinter as tk
from tkinter import font
from tkinter import messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from Manhatten import ManhattanSolver
from Bi_directional import BiDirectionalManhattanSolver
import random
import time
import tracemalloc
from IDA_star import IDAStarSolver
from Custom import CustomSolver
import sys
import numpy as np
from scipy.interpolate import make_interp_spline
import statistics


class MazeApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Maze Generator and Solver")
        self.size = 21  
        self.maze = [[1 for _ in range(self.size)] for _ in range(self.size)]  
        self.start = None
        self.end = None
        self.canvas = None
        self.fig, self.ax = None, None  # Initialize figure and axis 
        self.solver_instance = None  # To store current solver instance
        self.xlim = None
        self.ylim = None
        self.drag_start = None  # Store the starting point of dragging

        # Bind the close event
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.create_widgets()


    def on_close(self):
        """Handle the window close event to ensure proper cleanup."""
        # Close any matplotlib figures
        plt.close('all')

        # Destroy the Tkinter root
        self.root.destroy()

        # Exit the program
        sys.exit()

    def create_widgets(self):
        # Create frames
        self.canvas_frame = tk.Frame(self.root)
        self.canvas_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.control_frame = tk.Frame(self.root)
        self.control_frame.pack(side=tk.BOTTOM, pady=10)

        # Define a font for buttons and labels
        button_font = font.Font(family="Helvetica", size=10, weight="bold")
        label_font = font.Font(family="Helvetica", size=10)

        # Add a Reset Camera button
        self.reset_camera_button = tk.Button(
            self.control_frame, text="Reset Camera", command=self.reset_view,
            font=button_font, bg="#FF5722", fg="white", padx=10, pady=5
        )
        self.reset_camera_button.pack(side=tk.LEFT, padx=5)

        # Maze Size label and entry
        tk.Label(self.control_frame, text="Maze Size:", font=label_font).pack(side=tk.LEFT, padx=5)
        self.size_entry = tk.Entry(self.control_frame, width=5, font=label_font)
        self.size_entry.pack(side=tk.LEFT, padx=5)
        self.size_entry.insert(0, str(self.size))

        # Generate Maze button
        self.generate_button = tk.Button(
            self.control_frame, text="Generate Maze", command=self.update_maze_size,
            font=button_font, bg="#4CAF50", fg="white", padx=10, pady=5
        )
        self.generate_button.pack(side=tk.LEFT, padx=5)

        # Allow Maze Loops checkbox
        self.allow_loops_var = tk.BooleanVar()
        self.allow_loops_var.set(False)
        self.allow_loops_checkbox = tk.Checkbutton(
            self.control_frame, text="Allow Maze Loops", variable=self.allow_loops_var, font=label_font
        )
        self.allow_loops_checkbox.pack(side=tk.LEFT, padx=5)

        # Solve Maze button
        self.solve_button = tk.Button(
            self.control_frame, text="Solve Maze", command=lambda: self.solve_maze(self.heuristic_var.get()),
            font=button_font, bg="#2196F3", fg="white", padx=10, pady=5
        )
        self.solve_button.pack(side=tk.LEFT, padx=5)

        # Graph Size/Accuracy button
        self.graph_button = tk.Button(
            self.control_frame, text="Graph Size/Accuracy", command=lambda: self.graph_size_accuracy(self.heuristic_var.get()),
            font=button_font, bg="#FFC107", fg="black", padx=10, pady=5
        )
        self.graph_button.pack(side=tk.LEFT, padx=5)

        # Compare Solvers button
        self.solver_compare_button = tk.Button(
            self.control_frame, text="Compare Solvers", command=self.plot_comparison_chart,
            font=button_font, bg="#9C27B0", fg="white", padx=10, pady=5
        )
        self.solver_compare_button.pack(side=tk.LEFT, padx=5)

        # Reset button
        self.reset_button = tk.Button(
            self.control_frame, text="Reset", command=self.reset_solver,
            font=button_font, bg="#FFC107", fg="black", padx=10, pady=5, state=tk.DISABLED
        )
        self.reset_button.pack(side=tk.LEFT, padx=5)

        # Heuristic selection menu
        self.heuristic_var = tk.StringVar(value="A* with Manhattan heuristic")
        self.heuristic_menu = tk.OptionMenu(
            self.control_frame, self.heuristic_var, "A* with Manhattan heuristic", "Bi-directional A* with Manhattan heuristic" , "IDA* with Manhattan heuristic" , "Custom Solver"
        )
        self.heuristic_menu.config(font=label_font, bg="#f0f0f0", padx=5, pady=5)
        self.heuristic_menu.pack(side=tk.LEFT, padx=5)

    def reset_view(self):
        """Reset the zoom and pan to the default view."""
        if self.ax:
            self.ax.set_xlim(0, self.size)
            self.ax.set_ylim(self.size, 0)
            self.canvas.draw()

    def enable_zoom_and_drag(self):
        """Enable zooming and dragging on the maze plot."""
        self.fig.canvas.mpl_connect('scroll_event', self.on_zoom)
        self.fig.canvas.mpl_connect('button_press_event', self.on_drag_start)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_drag)
        self.fig.canvas.mpl_connect('button_release_event', self.on_drag_end)
    
    def on_zoom(self, event):
        """Handle zooming with the mouse scroll wheel."""
        if self.ax:
            zoom_factor = 1.2
            x_min, x_max = self.ax.get_xlim()
            y_min, y_max = self.ax.get_ylim()

            x_range = (x_max - x_min) / 2
            y_range = (y_max - y_min) / 2
            x_center = (x_min + x_max) / 2
            y_center = (y_min + y_max) / 2

            if event.button == 'up':  # Zoom in
                x_range /= zoom_factor
                y_range /= zoom_factor
            elif event.button == 'down':  # Zoom out
                x_range *= zoom_factor
                y_range *= zoom_factor

            # Flip the y-axis zoom
            self.ax.set_xlim(x_center - x_range, x_center + x_range)
            self.ax.set_ylim(y_center - y_range, y_center + y_range)
            self.canvas.draw()


    def on_drag_start(self, event):
        """Store the initial point when the mouse is clicked for dragging."""
        if self.ax and event.button == 1 and event.xdata is not None and event.ydata is not None:  # Check valid event
            self.drag_start = (event.xdata, event.ydata)

    def on_drag_end(self, event):
        """Reset drag_start when the mouse button is released."""
        if event.button == 1:  # Left mouse button
            self.drag_start = None

    def on_drag(self, event):
        """Handle dragging to pan the maze."""
        if self.ax and self.drag_start and event.xdata is not None and event.ydata is not None:  # Check valid event
            dx = event.xdata - self.drag_start[0]
            dy = event.ydata - self.drag_start[1]
            x_min, x_max = self.ax.get_xlim()
            y_min, y_max = self.ax.get_ylim()

            self.ax.set_xlim(x_min - dx, x_max - dx)
            self.ax.set_ylim(y_min - dy, y_max - dy)
            self.drag_start = (event.xdata, event.ydata)
            self.canvas.draw()

    def reset_solver(self):
        """Reset the current solver, keep the maze intact, and reset the visualization."""
        # Clear the solver instance
        self.solver_instance = None

        # Stop the
        
        # Clear the visualized path (reset the drawing on the canvas)
        if self.ax:
            self.ax.clear()  # Clear the existing plot
            self.ax.imshow(self.maze, cmap='binary')  # Redraw the maze
            self.ax.scatter(self.start[1], self.start[0], c='green', s=100)  # Start point
            self.ax.scatter(self.end[1], self.end[0], c='red', s=100)  # End point
            self.ax.set_xticks([])
            self.ax.set_yticks([])
            self.canvas.draw()  # Redraw the canvas

        # Re-enable the Solve button and disable the Reset button
        self.reset_button.config(state=tk.DISABLED)
        self.solve_button.config(state=tk.NORMAL)
        self.generate_button.config(state=tk.NORMAL)
        self.heuristic_menu.config(state=tk.NORMAL)

    def enable_controls(self):
        """Enable buttons after the solving process is complete."""
        self.generate_button.config(state=tk.NORMAL)
        self.solve_button.config(state=tk.NORMAL)
        self.heuristic_menu.config(state=tk.NORMAL)
        self.reset_button.config(state=tk.DISABLED)

    def disable_controls(self):
        """Disable buttons during the solving process."""
        self.generate_button.config(state=tk.DISABLED)
        self.solve_button.config(state=tk.DISABLED)
        self.heuristic_menu.config(state=tk.DISABLED)
        self.reset_button.config(state=tk.NORMAL)    


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
                solver = ManhattanSolver(self.maze, self.start, self.end , None)
            elif heuristic_var == "Bi-directional A* with Manhattan heuristic":
                solver = BiDirectionalManhattanSolver(self.maze, self.start, self.end , None)
            elif heuristic_var == "IDA* with Manhattan heuristic":
                solver = IDAStarSolver(self.maze , self.start , self.end , None)
            elif heuristic_var == "Custom Solver":
                solver = CustomSolver(self.maze , self.start , self.end , None)
            else:
                raise ValueError("Invalid heuristic option")

            tracemalloc.start()  # Start tracking memory
            start_time = time.time()  # Start timing

            path = solver.solve()

            end_time = time.time()  # End timing
            current, peak = tracemalloc.get_traced_memory()  # Get memory usage
            tracemalloc.stop()  # Stop tracking memory

            time_taken = end_time - start_time
            if heuristic_var == "IDA* with Manhattan heuristic":
                accuracy = solver.last_bound
            elif heuristic_var == "Custom Solver":
                print("Custom Solver")
                accuracy = (solver.end_last_bound + solver.start_last_bound)/2
            elif path is None:
                accuracy = 0    
            else:    
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
            "Bi-directional A* with Manhattan heuristic",
            "IDA* with Manhattan heuristic",
            "Custom Solver"
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
        plt.clf()
        plt.figure(figsize=(10, 6))

        # Plotting the bars for memory usage and time taken
        memory_bars = plt.bar([i - bar_width/2 for i in indices], avg_memory_usages, width=bar_width, label='Memory Usage (MB)')
        time_bars = plt.bar([i + bar_width/2 for i in indices], avg_times_taken, width=bar_width, label='Time Taken (s)')

       # Annotating the accuracy as a number in the middle of the bars
        for i, accuracy in enumerate(avg_accuracies):
            # Calculate the midpoint between the two bars
            midpoint = (avg_memory_usages[i] + avg_times_taken[i]) / 2
            if i != 2:
                plt.text(i, midpoint, f'Accuracy: {accuracy:.2%}', 
                    ha='center', va='center', fontsize=12, color='black')
            else:
                plt.text(i, midpoint, f'Avg Bound: {accuracy:.2f}', 
                    ha='center', va='center', fontsize=12, color='black')

        # clear the current figure
        plt.xticks(indices, solvers)
        plt.xlabel('Solvers')
        plt.ylabel('Values (AVG)')
        plt.title('Comparison of Solvers Approach (Large Search Space Mazes)')
        plt.legend()

        plt.show()

    def graph_size_accuracy(self, heuristic_var):
        print("Graphing size and accuracy")

        sizes = [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
        times = 50
        accuracies = []

        for size in sizes:
            for i in range(times):
                samples = []
                self.size = size
                self.generate_maze()

                if heuristic_var == "A* with Manhattan heuristic":
                    solver = ManhattanSolver(self.maze, self.start, self.end, None)
                elif heuristic_var == "Bi-directional A* with Manhattan heuristic":
                    solver = BiDirectionalManhattanSolver(self.maze, self.start, self.end, None)
                elif heuristic_var == "IDA* with Manhattan heuristic":
                    messagebox.showerror("Error", "This is only available for algorithms that use memory.")
                    return
                elif heuristic_var == "Custom Solver":
                    solver = CustomSolver(self.maze, self.start, self.end, None)
                else:
                    raise ValueError("Invalid heuristic option")

                path = solver.solve()

                if heuristic_var == "IDA* with Manhattan heuristic":
                    accuracy = solver.last_bound
                elif heuristic_var == "Custom Solver":
                    accuracy = (solver.end_last_bound + solver.start_last_bound) / 2
                elif path is None:
                    print("No solution found")
                    accuracy = 0
                else:
                    accuracy = len(path) / len(solver.explored_nodes) if solver.explored_nodes else 0

                samples.append(accuracy)
            mean_sample = statistics.mean(samples)
            accuracies.append(mean_sample)

        # Apply interpolation for smoother graph
        sizes_np = np.array(sizes)
        accuracies_np = np.array(accuracies)

        # Generate a spline for smooth interpolation
        spline = make_interp_spline(sizes_np, accuracies_np, k=3)  # Cubic spline
        smooth_sizes = np.linspace(sizes_np.min(), sizes_np.max(), 500)
        smooth_accuracies = spline(smooth_sizes)

        # Clear the current figure
        plt.clf()
        plt.plot(smooth_sizes, smooth_accuracies, label="Smoothed Accuracy")
        plt.scatter(sizes, accuracies, color='red', label="Original Data", zorder=5)  # Optional: show original points

        plt.xlabel("Maze Size")
        plt.ylabel("Accuracy")
        plt.title("Accuracy vs. Maze Size - " + heuristic_var)
        plt.legend()
        plt.show()

        
    
    
    
    
    def update_maze_size(self):
        try:
            size = int(self.size_entry.get())
            if size < 6:
                raise ValueError("Size must be greater than 6")
            if size % 2 == 0:
                size += 1  # Ensure size is odd , better for path finding
            self.size = size
            self.generate_maze()
        except ValueError:
            print("Invalid size input. Please enter a positive integer greater than 6.")

    def generate_maze(self):
        
        self.maze = [[1 for _ in range(self.size)] for _ in range(self.size)]  # Reset maze with walls
        # self.start = (0, 0)
        # self.end = (self.size - 1, self.size - 1)
        print ("Maze cleared")
        self.prim_maze_generation()
        print ("Maze primed")
        self.ensure_clear_start_end()
        print ("Start and end points ensured")
        if self.allow_loops_var.get():
            self.add_loops()
        self.draw_maze()



    def add_loops(self):
        
        
        potential_walls = []

        # Find all potential walls that can be removed
        for y in range(1, self.size - 1):  # Only check walls between passages
            for x in range(1, self.size - 1):
                if self.maze[y][x] == 1:
                    # Check if removing this wall would connect two paths (open spaces)
                    # Look at the two cells on either side of the wall
                    if (self.maze[y][x - 1] == 0 and self.maze[y][x + 1] == 0) or (self.maze[y - 1][x] == 0 and self.maze[y + 1][x] == 0):
                        potential_walls.append((x, y))


        # Randomly select walls to remove
        for wall in potential_walls:
            x, y = wall
            if random.random() < 0.1:
                self.maze[y][x] = 0
               
            




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

    def randomize_start_end(self):
        # Helper to get a random point on a specific wall
        def random_point_on_wall():
            wall = random.choice(["top", "bottom", "left", "right"])
            if wall == "top":
                return (0, random.randint(1, self.size - 2))  # Top row (0, x)
            elif wall == "bottom":
                return (self.size - 1, random.randint(1, self.size - 2))  # Bottom row
            elif wall == "left":
                return (random.randint(1, self.size - 2), 0)  # Left column (y, 0)
            elif wall == "right":
                return (random.randint(1, self.size - 2), self.size - 1)  # Right column


        # Randomize start and end points on the outer wall
        self.start = random_point_on_wall()

        # Ensure the end point is not the same as the start
        while True:
            self.end = random_point_on_wall()
            dist = abs(self.start[0] - self.end[0]) + abs(self.start[1] - self.end[1])
            print (dist)
            # if the manhatten distance from the start to the end is greater than the size of the maze
            if dist >= self.size:
                break

        sides = ["top", "bottom", "left", "right"]
        for side in sides:
            if side == "top":
                for x in range(0, self.size - 1):
                    self.maze[0][x] = 1
            elif side == "bottom":
                for x in range(0, self.size - 1):
                    self.maze[self.size - 1][x] = 1
            elif side == "left":
                for y in range(0, self.size - 1):
                    self.maze[y][0] = 1
            elif side == "right":
                for y in range(0, self.size - 1):
                    self.maze[y][self.size - 1] = 1

    def ensure_clear_start_end(self):
        """Clear paths around the start and end points in the maze."""
        self.randomize_start_end()
        print ("randomized start and end")
        # turn the start and end points into clear cells
        start_x, start_y = self.start
        self.maze[start_x][start_y] = 0
        end_x, end_y = self.end
        self.maze[end_x][end_y] = 0

        # clear all adjacent cells to the start and end points
        for x in range(start_x - 1, start_x + 2):
            for y in range(start_y - 1, start_y + 2):
                if 0 <= x < self.size and 0 <= y < self.size and self.maze[x][y] == 1:
                    self.maze[x][y] = 0
        for x in range(end_x - 1, end_x + 2):
            for y in range(end_y - 1, end_y + 2):
                if 0 <= x < self.size and 0 <= y < self.size and self.maze[x][y] == 1:
                    self.maze[x][y] = 0            
     


    def draw_maze(self):
        """Draw the maze and initialize zoom/drag functionality."""
        if not self.fig or not self.ax:
            self.fig, self.ax = plt.subplots(figsize=(6, 6))
            self.ax.imshow(self.maze, cmap='binary')
            self.ax.scatter(self.start[1], self.start[0], c='green', s=100)
            self.ax.scatter(self.end[1], self.end[0], c='red', s=100)
            self.ax.set_xticks([])
            self.ax.set_yticks([])

            self.canvas = FigureCanvasTkAgg(self.fig, master=self.canvas_frame)
            self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
            self.canvas.draw()

            # Enable zoom and drag after creating the plot
            self.enable_zoom_and_drag()

        else:
            self.ax.clear()
            self.ax.imshow(self.maze, cmap='binary')
            self.ax.scatter(self.start[1], self.start[0], c='green', s=100)
            self.ax.scatter(self.end[1], self.end[0], c='red', s=100)
            self.ax.set_xticks([])
            self.ax.set_yticks([])
            self.canvas.draw()

    def solve_maze(self , heuristic_var = "A* with Manhattan heuristic"):
        
        self.disable_controls()

        if heuristic_var == "A* with Manhattan heuristic":
            solver = ManhattanSolver(self.maze , self.start , self.end , self.draw_explored_path)
        elif heuristic_var == "Bi-directional A* with Manhattan heuristic":
            solver = BiDirectionalManhattanSolver(self.maze , self.start , self.end , self.draw_explored_path)
        elif heuristic_var == "IDA* with Manhattan heuristic":
            solver = IDAStarSolver(self.maze , self.start , self.end , self.draw_explored_path)
        elif heuristic_var == "Custom Solver":
            solver = CustomSolver(self.maze , self.start , self.end , self.draw_explored_path)           
        else :
            raise ValueError("invalid heuristic")

        self.solver_instance = solver

        path = solver.solve()

        if path:
            if heuristic_var == "IDA* with Manhattan heuristic":
                accuracy = solver.last_bound
                self.draw_explored_path(solver.explored_nodes , path , accuracy , show_last_bound = True)  # Draw the explored nodes after the solution
            elif heuristic_var == "Custom Solver":
                accuracy = (solver.end_last_bound + solver.start_last_bound)/2
                self.draw_explored_path(solver.explored_nodes , path , accuracy , show_last_bound = True)  # Draw the explored nodes after the solution
            else:    
                accuracy = len(path) / len(solver.explored_nodes) if solver.explored_nodes else 0
                self.draw_explored_path(solver.explored_nodes , path , accuracy)  # Draw the explored nodes after the solution
        else:
            print("No path found.")
            self.draw_explored_path(solver.explored_nodes)  # Still draw the explored nodes if no solution

    def draw_explored_path(self, explored_nodes, path=None, accuracy=0, show_last_bound=False):
        if not self.fig or not self.ax:
            self.fig, self.ax = plt.subplots(figsize=(6, 6))
            self.canvas = FigureCanvasTkAgg(self.fig, master=self.canvas_frame)
            self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.ax.clear()  # Clear only the content of the axes, not the whole figure

        # Draw the maze background
        self.ax.imshow(self.maze, cmap='binary')

        # Draw the explored nodes
        for node in explored_nodes:
            if path and node not in path:
                self.ax.scatter(node[1], node[0], c='yellow', s=50) 
            else:
                self.ax.scatter(node[1], node[0], c='blue', s=50)

        # Draw start and end points
        self.ax.scatter(self.start[1], self.start[0], c='green', s=50)  # Start point
        self.ax.scatter(self.end[1], self.end[0], c='red', s=50)  # End point

        # Display accuracy or last bound
        if show_last_bound:
            self.ax.text(0.5, -0.1, f"Last Bound: {accuracy}", ha='center', va='center', transform=self.ax.transAxes, fontsize=12, color='black')
        else:
            self.ax.text(0.5, -0.1, f"Accuracy: {accuracy:.2%}", ha='center', va='center', transform=self.ax.transAxes, fontsize=12, color='black')

        # Hide ticks
        self.ax.set_xticks([])
        self.ax.set_yticks([])

        # Redraw the canvas
        self.canvas.draw()
        self.root.update()


   





def quit(root):
    """Quit the application."""
    root.destroy()
    sys.exit()


def restart(root):
    """Restart the application."""
    root.destroy()
    root = tk.Tk()
    tk.Button(root, text="Quit", command=lambda root=root: quit(root)).pack()
    tk.Button(root, text="Restart", command=lambda root=root: restart(root)).pack()
    app = MazeApp(root)
    root.mainloop()



if __name__ == "__main__":
    root = tk.Tk()
    tk.Button(root, text="Quit", command=lambda root=root: quit(root)).pack()
    tk.Button(root, text="Restart", command=lambda root=root: restart(root)).pack()
    app = MazeApp(root)
    root.mainloop()



