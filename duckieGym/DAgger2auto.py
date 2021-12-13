import os

for i in range(0, 10):
    for j in range(1, 5):
        sr = "python3 testForAuto.py --map-name " + str(j) + str(i)
        os.system(sr)
