import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def get_value(t, old_value, t0_current_state, current_state, speed_rate, loop_time):
    if current_state == 0:
        angle = 2 * np.pi * (t - t0_current_state) / loop_time
        targetted_value = np.array([[np.cos(angle), np.sin(angle)]])
        new_value = speed_rate * targetted_value + (1 - speed_rate) * old_value
        noise = 1e-2 * np.random.multivariate_normal(np.array([0, 0]), np.array([[1, 0], [0, 1]]))
        q = (t - t0_current_state) // loop_time
        r = (t - t0_current_state) % loop_time
        if r % (loop_time / 3) == 0:
            change = np.sum(np.random.randint(0, 2, 10)) > 8 - q
            if change:
                new_state = (r // (loop_time / 3)) + 1
                t0_new_state = t
                return new_value + noise, new_state, t0_new_state
        return new_value + noise, current_state, t0_current_state
    elif current_state == 1:
        angle = np.random.uniform(-np.pi, np.pi)
        dist = np.random.uniform(0, 1)
        targetted_value = np.array([[3, 0]]) + dist * np.array([[np.cos(angle), np.sin(angle)]])
        new_value = speed_rate * targetted_value + (1 - speed_rate) * old_value
        noise = 1e-2 * np.random.multivariate_normal(np.array([0, 0]), np.array([[1, 0], [0, 1]]))
        change = np.sum(np.random.randint(0, 2, 1000)) > 800 - (t - t0_current_state)
        if change:
            new_state = 0
            t0_new_state = t
            return new_value + noise, new_state, t0_new_state
        return new_value + noise, current_state, t0_current_state
    elif current_state == 2:
        targetted_value = np.array([[3 * -1 / 2, 3 * np.sqrt(3) / 2]]) + np.random.uniform(-1, 1, (1, 2))
        new_value = speed_rate * targetted_value + (1 - speed_rate) * old_value
        noise = 1e-2 * np.random.multivariate_normal(np.array([0, 0]), np.array([[1, 0], [0, 1]]))
        change = np.sum(np.random.randint(0, 2, 1000)) > 800 - (t - t0_current_state)
        if change:
            new_state = 0
            t0_new_state = t - (loop_time / 3)
            return new_value + noise, new_state, t0_new_state
        return new_value + noise, current_state, t0_current_state
    elif current_state == 3:
        targetted_value = np.random.uniform(-1, 1, (1, 2))
        targetted_value[0, 1] = (np.sqrt(3) / 2) * targetted_value[0, 1]
        if (targetted_value[0, 1] > np.sqrt(3) / 2 - np.sqrt(3) * targetted_value[0, 0]):
            targetted_value[0, 0] = targetted_value[0, 0] - (1 / 2)
            targetted_value[0, 1] = targetted_value[0, 1] - (np.sqrt(3) / 4)
        elif (targetted_value[0, 1] > np.sqrt(3) / 2 + np.sqrt(3) * targetted_value[0, 0]):
            targetted_value[0, 0] = targetted_value[0, 0] + (1 / 2)
            targetted_value[0, 1] = targetted_value[0, 1] - (np.sqrt(3) / 4)
        targetted_value = targetted_value + np.array([[3 * -1 / 2, 3 * -1 * np.sqrt(3) / 2]])
        new_value = speed_rate * targetted_value + (1 - speed_rate) * old_value
        noise = 1e-2 * np.random.multivariate_normal(np.array([0, 0]), np.array([[1, 0], [0, 1]]))
        change = np.sum(np.random.randint(0, 2, 1000)) > 800 - (t - t0_current_state)
        if change:
            new_state = 0
            t0_new_state = t - (2 * loop_time / 3)
            return new_value + noise, new_state, t0_new_state
        return new_value + noise, current_state, t0_current_state
    else:
        raise ValueError("'current_state' should be one of 0, 1, 2 or 3.")


if __name__ == "__main__":
    state = 0
    t0_state = 0
    size = 200000
    speed_rate = 0.5
    loop_time = 720
    values = np.zeros((size, 2))
    labels = np.zeros((size, 1))
    previous_value = np.array([[1, 0]])
    for t in range(size):
        values[t], state, t0_state = get_value(t, previous_value, t0_state, state, speed_rate, loop_time)
        previous_value = values[t].copy().reshape(1, 2)
        outlier = np.random.binomial(1, 5e-4)
        if outlier == 1:
            values[t] = np.random.uniform(-4, 4, (1, 2))
            labels[t] = -1
        else:
            labels[t] = 1
    df = pd.DataFrame(values, columns=['x1', 'x2'])
    df["y"] = labels
    # df.to_csv("../res/new.csv")
    ax = df.plot.scatter(x='x1', y='x2', c='y', alpha=0.1, colormap='viridis')
    ax.set_xlim((-4, 4))
    ax.set_ylim((-4, 4))
    ax.get_figure().set_figheight(16)
    ax.get_figure().set_figwidth(16)
    plt.show()
