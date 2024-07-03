import matplotlib.pyplot as plt
import numpy as np

def get_bitmap(direction = "r", xtess = 2, x = 720, t = 500, output_range = (-1, 1)) -> np.ndarray:
    bitmap = np.zeros((t, x))

    if x%(xtess) == 0:
        xstep = (int)(x/xtess)
        if direction == "r":
            for xiter in range(xtess):
            
                bitmap[:, xiter*xstep:(xiter+1)*xstep] = np.meshgrid(np.zeros(t),(np.linspace(0, 1, xstep)),  indexing = 'ij')[1]
        elif direction == "l":
            for xiter in range(xtess):
            
                bitmap[:, xiter*xstep:(xiter+1)*xstep] = np.meshgrid(np.zeros(t),(np.linspace(1, 0, xstep)),  indexing = 'ij')[1]
        else:
            print("Enter l or r for the direction.")
        
        center = (output_range[0] + output_range[1])/2
        spread = output_range[1]-output_range[0]

        bitmap *= spread
        bitmap = bitmap + center - (0.5 * spread)

        return bitmap

    else:
        print("Make sure x is divisible by xtess *shape[1] and ytess is divisible by ttess * shape[0]")
        return