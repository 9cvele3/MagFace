import numpy as np
import matplotlib.pyplot as plt

matrix = np.zeros((1, 512))

with open('MagFace/eval/eval_recognition/features/magface_iresnet18/lfw_features.list', 'r') as features:
    while True:
        line = features.readline()

        if not line:
            break

        fields = line.rstrip().split(' ')
        row = np.array([])

        ind = 0
        for f in fields:
            if ind > 0:
                row = np.append(row, 255 * float(f))

            ind += 1

        matrix = np.vstack([matrix, row])

print(matrix.shape)
plt.matshow(matrix)
plt.show()






