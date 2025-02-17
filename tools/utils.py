import numpy as np
import matplotlib.pyplot as plt


def plot_learning_curve(x, avg_scores, epsilons, filename):
    fig = plt.figure()
    ax = fig.add_subplot(111, label="1")
    ax2 = fig.add_subplot(111, label="2", frame_on=False)

    ax.plot(x, epsilons, color="C0")
    ax.set_xlabel("Trainig Steps", color="C0")
    ax.set_ylabel("Epsilon", color="C0")
    ax.tick_params(axis="x", colors="C0")
    ax.tick_params(axis="y", colors="C0")

    if len(x) > len(avg_scores):
        x = x[:len(avg_scores)]

    ax2.plot(x, avg_scores, color="C1")
    ax2.axes.get_xaxis().set_visible(False)
    ax2.yaxis.tick_right()
    ax2.set_ylabel("Score", color="C1")
    ax2.yaxis.set_label_position("right")
    ax2.tick_params(axis="y", colors="C1")

    plt.savefig("results/" + filename)


def get_one_hot(size, idx):
    one_hot = np.zeros(size)
    one_hot[idx] = 1
    return one_hot


def calculate_epsilon_decay(n_episodes, epsilon_min=0.01, initial_epsilon=1.0):
    return (epsilon_min / initial_epsilon) ** (1 / n_episodes)
