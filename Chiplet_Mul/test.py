import numpy as np
from matplotlib import pyplot as plt


def plot_simulation_comparison(time_fun, time_timing, cycle_fun, cycle_timing):
    # Create a new figure and set of subplots with adjusted size
    fig, ax1 = plt.subplots(figsize=(10, 8))

    # Set the figure title
    fig.suptitle('Simulation Time and Cycles Comparison', fontsize=16)

    # Set the data
    labels = ['Function Simulation', 'Timing Simulation']
    time_data = [round(time_fun, 2), round(time_timing, 2)]
    cycle_data = [cycle_fun, cycle_timing]

    # Set the positions
    x = np.arange(len(labels))  # the label locations

    # Define the width of the bars
    width = 0.4  # the width of the bars

    # Create two bar charts with different colors
    rects1 = ax1.bar(x - width/2, time_data, width, label='Time (seconds)', color='skyblue')
    ax1.set_ylabel('Time (seconds)', fontsize=14)
    ax1.set_ylim(0, max(time_data) * 1.2)  # Set y-axis upper limit to 1.2 times of max time for label space

    # Use the same x-axis but different y-axis
    ax2 = ax1.twinx()
    rects2 = ax2.bar(x + width/2, cycle_data, width, label='Cycles (count)', color='lightgreen')
    ax2.set_ylabel('Cycles (count)', fontsize=14)
    ax2.set_ylim(0, max(cycle_data) * 1.2)  # Set y-axis upper limit to 1.2 times of max cycles for label space

    # Add data labels
    def autolabel(rects, ax):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=10)

    autolabel(rects1, ax1)
    autolabel(rects2, ax2)

    # Set x-axis labels
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=14)
    ax1.set_xlabel('Simulation Type', fontsize=14)

    # Set legend position
    fig.legend(loc='upper center', bbox_to_anchor=(0.5, 0.95), ncol=2, fontsize=12)

    # Adjust layout
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])  # Leave space at the top for the title and legend

    # Show the figure
    plt.show()
plot_simulation_comparison(1,2,3,4)