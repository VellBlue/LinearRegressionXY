import tkinter as tk
from tkinter import messagebox
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


class LinearRegressionApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Linear Regression")

        # Frame for x data
        self.frame_x = tk.Frame(master)
        self.frame_x.grid(row=0, column=0, padx=10, pady=10)
        self.x_label = tk.Label(self.frame_x, text="Data for x:")
        self.x_label.pack()
        self.x_entry = tk.Text(self.frame_x, height=10, width=25)
        self.x_entry.pack()

        # Frame for y data
        self.frame_y = tk.Frame(master)
        self.frame_y.grid(row=0, column=1, padx=10, pady=10)
        self.y_label = tk.Label(self.frame_y, text="Data for y:")
        self.y_label.pack()
        self.y_entry = tk.Text(self.frame_y, height=10, width=25)
        self.y_entry.pack()

        # Frame for buttons
        self.frame_buttons = tk.Frame(master)
        self.frame_buttons.grid(row=1, column=0, columnspan=2, pady=(0, 10))
        self.calculate_button = tk.Button(self.frame_buttons, text="Calculate Linear Regression",
                                          command=self.calculate)
        self.calculate_button.pack(side=tk.LEFT, padx=(0, 5))
        self.plot_button = tk.Button(self.frame_buttons, text="Show Graph", command=self.plot)
        self.plot_button.pack(side=tk.LEFT)

        # Frame for results
        self.frame_results = tk.Frame(master)
        self.frame_results.grid(row=2, column=0, columnspan=2, padx=10, pady=10)
        self.result_label = tk.Label(self.frame_results, text="")
        self.result_label.pack()

    def calculate(self):
        x_data = self.x_entry.get("1.0", tk.END)
        y_data = self.y_entry.get("1.0", tk.END)

        try:
            x_data = x_data.strip().split('\n')
            x_data = [float(d) for d in x_data]
            y_data = y_data.strip().split('\n')
            y_data = [float(d) for d in y_data]
        except ValueError:
            messagebox.showerror("Error", "Enter numerical values separated by spaces.")
            return

        if len(x_data) != len(y_data):
            messagebox.showerror("Error", "The number of data points for x and y must be the same.")
            return

        if len(x_data) < 2:
            messagebox.showerror("Error", "Enter at least two data points.")
            return

        x = np.array(x_data).reshape(-1, 1)
        y = np.array(y_data)

        correlation_matrix = np.corrcoef(x.flatten(), y)
        correlation_coefficient = correlation_matrix[0, 1]

        model = LinearRegression()
        model.fit(x, y)

        slope = model.coef_[0]
        intercept = model.intercept_

        self.result_label.config(
            text=f"Correlation (r): {correlation_coefficient:.2f}\nEquation of the line: y = {slope:.2f}x + {intercept:.2f}")

    def plot(self):
        x_data = self.x_entry.get("1.0", tk.END)
        y_data = self.y_entry.get("1.0", tk.END)

        try:
            x_data = x_data.strip().split('\n')
            x_data = [float(d) for d in x_data]
            y_data = y_data.strip().split('\n')
            y_data = [float(d) for d in y_data]
        except ValueError:
            messagebox.showerror("Error", "Enter numerical values separated by spaces.")
            return

        if len(x_data) != len(y_data):
            messagebox.showerror("Error", "The number of data points for x and y must be the same.")
            return

        if len(x_data) < 2:
            messagebox.showerror("Error", "Enter at least two data points.")
            return

        x = np.array(x_data).reshape(-1, 1)
        y = np.array(y_data)

        model = LinearRegression()
        model.fit(x, y)

        plt.scatter(x, y, color='blue')
        plt.plot(x, model.predict(x), color='red')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Linear Regression')
        plt.show()


def main():
    root = tk.Tk()
    app = LinearRegressionApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
