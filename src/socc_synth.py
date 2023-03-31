#%%
import numpy as np
import matplotlib.pyplot as plt
import matplotsoccer as mps
from mplsoccer import Pitch

# Total amount of sequences
TOTAL=1000

# Interpolate points between each point
# num_points is the number of points to interpolate between each point
def interpolate_points(t,x, y, num_points=10):
    # Create evenly spaced intervals between each point
    intervals = np.linspace(0, 1, num_points+1,endpoint=False)[:-1]
    
    # Interpolate the x and y values at each interval
    interp_t = []
    interp_x = []
    interp_y = []
    for i in range(len(x)-1):
        t_vals = t[i:i+2]
        x_vals = x[i:i+2]
        y_vals = y[i:i+2]
        interp_t.extend(np.interp(intervals, [0, 1], t_vals))
        interp_x.extend(np.interp(intervals, [0, 1], x_vals))
        interp_y.extend(np.interp(intervals, [0, 1], y_vals))

    
        
    return interp_t,interp_x, interp_y



# Generate synthetic data with gaussian distribution
def gaussian_data(l=6,t=[0,2,5,9,14,20,26,33],x=[10,20,30,60,90,100,70,80],y=[8,60,10,60,10,60,56,64]):

    n = int(TOTAL)
    # Make data length
    t = t[:l]
    x = x[:l]
    y = y[:l]

    all_t = []
    all_x = []
    all_y = []


    # x_std = .05 / 2
    x_std = 3 / l
    y_std = x_std * 68 / 105
    t_std = 3

    # Create gaussian distribution
    for i in range(l):
        t1 = np.random.normal(t[i], t_std, TOTAL)
        x1 = np.random.normal(x[i], x_std*((i+1)), TOTAL)
        y1 = np.random.normal(y[i], y_std*((i+1)), TOTAL)
        # Put all t in a column
        all_t.append(t1.reshape(-1, 1))
        # Put all x in a column
        all_x.append(x1.reshape(-1, 1))
        # Put all y in a column
        all_y.append(y1.reshape(-1, 1))

    all_data = np.empty((n, l, 3))
    
    # DUMB WAY TO DO IT
    # Concatenate all the t arrays in all_t
    t = np.concatenate(all_t, axis=1)
    # make sure they're less than 45
    t[t > 45] = 65
    # make sure they're greater than 0
    t[t < 0] = 0
    # Make sure they're in ascending order for each array
    t = np.sort(t, axis=1)
    # Make sure each subarray has different values
    for i in range(len(t)):
        # If there are any duplicates in values add a delta to the second value
        if len(np.unique(t[i])) != len(t[i]):
            t[i][1] += 1
    # Make sure the array is still in ascending order
    t = np.sort(t, axis=1)

    

    all_data[:, :, 0] = t
    # Concatenate all the x arrays in all_x
    x = np.concatenate(all_x, axis=1)
    # make sure they're less than 105
    x[x > 105] = 105
    # make sure they're greater than 0
    x[x < 0] = 0
    all_data[:, :, 1] = x
    # Concatenate all the y
    y = np.concatenate((all_y), axis=1)
    # make sure they're less than 68 in value
    y[y > 68] = 68
    # make sure they're greater than 0 in value
    y[y < 0] = 0
    all_data[:, :, 2] = y

    return all_data
#%%
def display(data,color="blue"):
    mps.field("green",figsize=8, show=False)
    data = data[:,:,1:]
    # drop the 4th row of each array
    # data = data[:, :-1, :]
    # x is the first column of each array
    x = data[:,:,0]
    # y is the second column of each array
    y = data[:,:,1]
    # Subtract 68 from y to flip the y axis
    l = data.shape[1]
    print(l)
    # plt.scatter(x,y,color=color)
    data1 = data[:, :, :]
    # data2 is the last row of each array
    data2 = data[:, -1, :]
    x1 = data1[:,:,0]
    y1 = data1[:,:,1]
    plt.scatter(x1,y1,s=0.01,alpha=0.7,color="black")
    x2 = data2[:,0]
    y2 = data2[:,1]
    plt.scatter(x2,y2,s=0.01,alpha=0.7,color="red")
    plt.axis("on")
    plt.show()

# Create main to run the code
if __name__ == '__main__':
    
    
    # Center of gaussian distribution
    t = [0,5,12,22,30,42]
    x = [10,20,30,40,50,60]
    y = [10,20,30,40,50,60]

    # UNCOMMENT TO INTERPOLATE
    # t,x,y = interpolate_points(t,x,y)

    l = len(t)
    data = gaussian_data(l=l,t=t,x=x,y=y)

    display(data)

    # Save data
    np.savez("./data/interim/data_seq_socc.npz", arr_0=data)

    

# %%
