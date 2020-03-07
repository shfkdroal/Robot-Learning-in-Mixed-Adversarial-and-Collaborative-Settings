
import numpy as np
import scipy.io

min_coord = 105
range__ = 285


def average_val_R_table_deck(deck):
    sum_reward = 0
    sum_angle = 0

    for e in deck:
        sum_reward += e[0]
        sum_angle += e[1]
    return (sum_reward / len(deck), sum_angle / len(deck))

def Approximate_R_value(X, Y, R_table):

    X = int(X)
    Y = int(Y)

    minnum = int(min_coord)
    range_ = range__
    maxnum = minnum + range_ - 1
    N = 1
    deck = []
    approximate_R_val = None

    x = None
    y = None

    min_phase_num = 20

    while len(deck) == 0 or min_phase_num >= 0:
        y = Y + (N - 1)
        if y >= maxnum:
            y = maxnum
        for ind in range(X - (N - 1), X + (N - 1) + 1):
            x = ind
            if x <= minnum:
                x = minnum
            elif x >= maxnum:
                x = maxnum
            if R_table[x - minnum, y - minnum] != (-1, -1):
                deck.append(R_table[x - minnum, y - minnum])
        x = X - (N - 1)
        if x <= minnum:
            x = minnum
        for ind in range(Y - (N - 1), Y + (N - 1) + 1):
            y = ind
            if y <= minnum:
                y = minnum
            elif y >= maxnum:
                y = maxnum
            if R_table[x - minnum, y - minnum] != (-1, -1):
                deck.append(R_table[x - minnum, y - minnum])
        y = Y - (N - 1)
        if y <= minnum:
            y = minnum
        for ind in range(X - (N - 1), X + (N - 1) + 1):
            x = ind
            if x <= minnum:
                x = minnum
            elif x >= maxnum:
                x = maxnum
            if R_table[x - minnum, y - minnum] != (-1, -1):
                deck.append(R_table[x - minnum, y - minnum])
        x = X + (N - 1)
        if x >= maxnum:
            x = maxnum
        for ind in range(Y - (N - 1), Y + (N - 1) + 1):
            y = ind
            if y <= minnum:
                y = minnum
            elif y >= maxnum:
                y = maxnum
            if R_table[x - minnum, y - minnum] != (-1, -1):
                deck.append(R_table[x - minnum, y - minnum])

        N += 1
        min_phase_num -= 1

    return average_val_R_table_deck(deck)



f = open("./gt_center_avg_reward.txt", "r")

value = np.empty((), dtype=object)
range_ = 285
value[()] = (-1, -1)  # reward - angle pairs
R_table = np.full([range_, range_], value, dtype=object)
R_table_full = np.full([range_, range_], value, dtype=object)
range_min = 105

xcoord = 0
ycoord = 0

while True:
    line = f.readline()
    if not line:
        break
    if line[0] == "[":
        xcoord = int(line.split(", ")[0].split("[")[1])
        ycoord = int(line.split(", ")[1].split("]")[0])
    else:
        r = float(line)
        R_table[xcoord - range_min, ycoord - range_min] = (r, 0)


f.close()

print(R_table)

for i in range(285):
    for j in range(285):
        if R_table[i, j][0] == -1.0:
            R_table_full[i, j] = Approximate_R_value(i + min_coord, j + min_coord, R_table)
        else:
            R_table_full[i, j] = R_table[i, j]


print("R_table_gt: ", R_table_full)
np.save("./R_table_gt.npy", R_table_full)
scipy.io.savemat('./R_table_gt.mat', {'R_table_full': R_table_full})