import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np


class PlotHelper:
    def __init__(self, fig_size: tuple = (10, 8)) -> Figure:
        """
        Wrapper class to simplify the plotting for different results.
        Parameters:
        -----------
        fig_size: tuple, optional
            Size of the figure(s), defaults to (10, 8)
        """
        self.fig_size = fig_size
        self.markers = ["o", "x", "^", "s", "*"]        # assuming we'd never need more than these

    def plot_data(
        self,
        data: tuple[np.ndarray],
        legend_labels: tuple[str],
        title: str,
        xlabel: str = "",
        ylabel: str = "",
    ) -> None:
        """
        Plots the data and sets the properties of the figures.
        Parameters:
        -----------
        data: tuple[np.ndarray]
            A sequence data to be plotted. All of them get recursively plotted on the same figure.
        legend_labels: tuple[str]
            Sequence of labels corresponding to each array in data.
        title: str
            Title for the figure.
        xlabel, ylabel: str, optional
            Labels for the x- and y-axis. Defaults to empty strings.

        Returns:
        --------
        Matplotlib Figure object. 
        """
        fig_obj = plt.figure(figsize=self.fig_size)
        for index, datum in enumerate(data):
            plt.plot(
                datum[0, :],
                datum[1, :],
                marker=self.markers[index],
                label=legend_labels[index],
            )

        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend()
        plt.grid(True)

        return fig_obj


def present_results(
    player_trajectory: np.ndarray,
    estimated_trajectory: np.ndarray,
    delta_t: float,
    error_plot: bool,
    velocity_plot: bool,
):
    """
    Function that can present some results based on the actual and estimated trajectory.
    Computes the error norm and the velocity and plots the results.

    Parameters:
    -----------
    player_trajectory: np.ndarray
        History of the true positions of the player.
    estimated_trajectory: np.ndarray
        Estimated positions of the player
    delta_t: float
        Time between consecutive measurements, used for velocity calculation.
    error_plot, velocity_plot: bool
        Show or hide plots related to error norm, or velocity.
    """
    plot_handle = PlotHelper(fig_size=(10, 8))

    # plot actual and estimated values on one figure
    est_comparision = [player_trajectory[0:2, :], estimated_trajectory[0:2, :]]
    est_labels = ("Player trajectory", "Estimated trajectory")
    est_title = "Results of player position estimates"
    plot_handle.plot_data(est_comparision, est_labels, est_title)

    # same label for two figures
    x_time_label = "Time step"

    if error_plot:
        total_iterations = player_trajectory.shape[1]
        err_iterations = np.arange(total_iterations)

        # compute error norm, stack with counter
        error = np.linalg.norm(player_trajectory - estimated_trajectory[0:2, :], axis=0)
        error_itr_stacked = np.vstack((err_iterations,error))

        error_legend = ("Normed error",)
        error_title = "Norm of difference between positions"
        error_ylabel = "Normed error"
        plot_handle.plot_data((error_itr_stacked,), error_legend, error_title, x_time_label, error_ylabel)
    
    if velocity_plot:
        # compute velocity, stack with counter
        velocity = np.diff(estimated_trajectory, axis=1) * delta_t
        velocity = np.linalg.norm(velocity, axis=0)
        vel_iterations = np.arange(velocity.shape[0])
        velocity_itr_stacked = np.vstack((vel_iterations, velocity))

        velocity_legend = ("Velocity",)
        velocity_ylabel = "Velocity m/s"
        velocity_title = "Estimated velocity of the player"
        plot_handle.plot_data((velocity_itr_stacked,), velocity_legend, velocity_title, x_time_label, velocity_ylabel)

    plt.show()