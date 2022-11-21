from dsod.utils import RStarTreeObject
import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    square_to_clip = RStarTreeObject(np.array([[0, 0]], dtype='float64'), np.array([[5, 5]], dtype='float64'))
    # q = RStarTreeObject(np.array([[5, 1]], dtype='float64'), np.array([[5, 1]], dtype='float64'))
    q = RStarTreeObject(np.array([[10, 0]], dtype='float64'), np.array([[5, 1]], dtype='float64'))
    p_1 = RStarTreeObject(np.array([[3, 3]], dtype='float64'), np.array([[3, 3]], dtype='float64'))
    p_2 = RStarTreeObject(np.array([[5.1, 2.5]], dtype='float64'), np.array([[5.1, 2.5]], dtype='float64'))

    square_1 = square_to_clip.copy()
    square_1 = square_1.__clip__(q, [p_1], 3)
    if square_1.contains_nan():
        square_1.low = np.array([[0, 0]], dtype='float64')
        square_1.high = np.array([[0, 0]], dtype='float64')
    square_2 = square_to_clip.copy()
    square_2 = square_2.__clip__(q, [p_2], 3)
    if square_2.contains_nan():
        square_2.low = np.array([[0, 0]], dtype='float64')
        square_2.high = np.array([[0, 0]], dtype='float64')
    square_1_2 = square_to_clip.copy()
    square_1_2 = square_1_2.__clip__(q, [p_1, p_2], 3)
    if square_1_2.contains_nan():
        square_1_2.low = np.array([[0, 0]], dtype='float64')
        square_1_2.high = np.array([[0, 0]], dtype='float64')

    fig, ax = plt.subplots()

    s_original = plt.Rectangle((square_to_clip.low[0, 0], square_to_clip.low[0, 1]), square_to_clip.high[0, 0] - square_to_clip.low[0, 0],
                               square_to_clip.high[0, 1] - square_to_clip.low[0, 1], color='k', fill=False)
    s_1 = plt.Rectangle((square_1.low[0, 0], square_1.low[0, 1]), square_1.high[0, 0] - square_1.low[0, 0],
                               square_1.high[0, 1] - square_1.low[0, 1], linestyle=(0, (3, 9)), color='orange', fill=False)
    s_2 = plt.Rectangle((square_2.low[0, 0], square_2.low[0, 1]), square_2.high[0, 0] - square_2.low[0, 0],
                               square_2.high[0, 1] - square_2.low[0, 1], linestyle=(0, (3, 9)), color='green', fill=False)
    s_1_2 = plt.Rectangle((square_1_2.low[0, 0], square_1_2.low[0, 1]), square_1_2.high[0, 0] - square_1_2.low[0, 0],
                               square_1_2.high[0, 1] - square_1_2.low[0, 1], linestyle=(0, (3, 9)), color='blue', fill=False)

    ax.add_patch(s_original)
    ax.add_patch(s_1)
    ax.add_patch(s_2)
    ax.add_patch(s_1_2)
    ax.scatter(q.low[0, 0], q.low[0, 1], marker='+', color='k')
    ax.scatter(p_1.low[0, 0], p_1.low[0, 1], marker='+', color='orange')
    ax.scatter(p_2.low[0, 0], p_2.low[0, 1], marker='+', color='green')

    plt.show()
