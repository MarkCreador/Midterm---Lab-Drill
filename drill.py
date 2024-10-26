import numpy as np
import tkinter as tk
from collections import deque
import heapq
import time

class MazeSolverApp:
    def __init__(self, root):
        self.root = root
        self.root.title("aMaze Solver")
        self.maze = np.array([
            [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0],
            [0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0],
            [0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0],
            [0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0],
            [0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0],
            [1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0],
            [0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0],
            [0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        ])
        self.start = (1, 1)  
        self.end = (12, 12)  

        self.build_ui()

    def build_ui(self):
     
        self.maze_canvas = tk.Canvas(self.root, width=250, height=250, bg="white")
        self.maze_canvas.grid(row=0, column=0, columnspan=3)
        self.draw_maze()

       
        self.algo_var = tk.StringVar(value="A*")
        tk.Radiobutton(self.root, text="BFS", variable=self.algo_var, value="BFS").grid(row=1, column=0)
        tk.Radiobutton(self.root, text="DFS", variable=self.algo_var, value="DFS").grid(row=1, column=1)
        tk.Radiobutton(self.root, text="A*", variable=self.algo_var, value="A*").grid(row=1, column=2)
        tk.Button(self.root, text="Solve Maze", command=self.solve_maze).grid(row=2, column=1)
        tk.Button(self.root, text="Reset Maze", command=self.reset_maze).grid(row=3, column=1)

    def draw_maze(self):
        self.maze_canvas.delete("all")  
        rows, cols = self.maze.shape
        cell_width = 250 / cols
        cell_height = 250 / rows

        for i in range(rows):
            for j in range(cols):
                
                color = "white" if self.maze[i, j] == 1 else "black"
                self.maze_canvas.create_rectangle(
                    j * cell_width, i * cell_height, (j + 1) * cell_width, (i + 1) * cell_height, fill=color
                )
      
        self.maze_canvas.create_oval(5, 5, 25, 25, fill="green")  
        self.maze_canvas.create_oval(225, 225, 245, 245, fill="red")  

    def reset_maze(self):
        """Reset the maze to its initial state."""
        self.path = []  
        self.draw_maze()  
        self.maze_canvas.update() 

    def solve_maze(self):
        solver = MazeSolver(self.maze, self.start, self.end, self.maze_canvas)
        algorithm = self.algo_var.get()

        if algorithm == "BFS":
            solver.solve_bfs(animated=True)
        elif algorithm == "DFS":
            solver.solve_dfs(animated=True)
        elif algorithm == "A*":
            solver.solve_a_star(animated=True)

class MazeSolver:
    def __init__(self, maze, start, end, canvas):
        self.maze = maze
        self.start = start
        self.end = end
        self.canvas = canvas
        self.path = []

    def solve_bfs(self, animated=False):
        """Solve the maze using Breadth-First Search."""
        rows, cols = self.maze.shape
        queue = deque([self.start])
        visited = set()
        visited.add(self.start)
        prev = {self.start: None}

        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)] 

        while queue:
            current = queue.popleft()
            if current == self.end:
                self.path = self._reconstruct_path(prev)
                if animated:
                    self.animate_path()
                return True

            for direction in directions:
                neighbor = (current[0] + direction[0], current[1] + direction[1])
                if (0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols and 
                    self.maze[neighbor] == 0 and neighbor not in visited):
                    visited.add(neighbor)
                    queue.append(neighbor)
                    prev[neighbor] = current
                    if animated:
                        self.animate_cell(neighbor)

        return False

    def solve_dfs(self, animated=False):
        """Solve the maze using Depth-First Search."""
        rows, cols = self.maze.shape
        stack = [self.start]
        visited = set()
        visited.add(self.start)
        prev = {self.start: None}

        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  

        while stack:
            current = stack.pop()
            if current == self.end:
                self.path = self._reconstruct_path(prev)
                if animated:
                    self.animate_path()
                return True

            for direction in directions:
                neighbor = (current[0] + direction[0], current[1] + direction[1])
                if (0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols and 
                    self.maze[neighbor] == 0 and neighbor not in visited):
                    visited.add(neighbor)
                    stack.append(neighbor)
                    prev[neighbor] = current
                    if animated:
                        self.animate_cell(neighbor)

        return False

    def solve_a_star(self, animated=False):
        """Solve the maze using A* Algorithm."""
        rows, cols = self.maze.shape
        priority_queue = []
        heapq.heappush(priority_queue, (0, self.start))
        cost_from_start = {self.start: 0}
        prev = {self.start: None}

        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  
        def h(cell):
            """Heuristic function (Manhattan distance)"""
            return abs(cell[0] - self.end[0]) + abs(cell[1] - self.end[1])

        while priority_queue:
            _, current = heapq.heappop(priority_queue)
            if current == self.end:
                self.path = self._reconstruct_path(prev)
                if animated:
                    self.animate_path()
                return True

            for direction in directions:
                neighbor = (current[0] + direction[0], current[1] + direction[1])
                if (0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols and 
                    self.maze[neighbor] == 0):
                    new_cost = cost_from_start[current] + 1
                    if (neighbor not in cost_from_start or new_cost < cost_from_start[neighbor]):
                        cost_from_start[neighbor] = new_cost
                        priority = new_cost + h(neighbor)
                        heapq.heappush(priority_queue, (priority, neighbor))
                        prev[neighbor] = current
                        if animated:
                            self.animate_cell(neighbor)

        return False

    def _reconstruct_path(self, prev):
        """Reconstruct the path from the previous cell mapping."""
        path = []
        current = self.end
        while current is not None:
            path.append(current)
            current = prev[current]
        return path[::-1]  

    def animate_cell(self, cell):
        """Animate exploration of a single cell"""
        cell_width = 250 / self.maze.shape[1]
        cell_height = 250 / self.maze.shape[0]
        x_center = cell[1] * cell_width + cell_width / 2
        y_center = cell[0] * cell_height + cell_height / 2
        self.canvas.create_oval(
            x_center - 5, y_center - 5, x_center + 5, y_center + 5, fill="yellow"
        )
        self.canvas.update()
        time.sleep(0.2)  

    def animate_path(self):
        """Animate the final path"""
        for cell in self.path[1:-1]:  
            self.animate_cell(cell)

if __name__ == "__main__":
    root = tk.Tk()
    app = MazeSolverApp(root)
    root.mainloop()
