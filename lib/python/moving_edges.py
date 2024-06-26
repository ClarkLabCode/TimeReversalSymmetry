#May be worth it to clean up this code by turning each stimulus into a class?
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
import random
import time
import itertools
import math
from skimage.transform import resize, downscale_local_mean

import imageio
import matplotlib.image as image

#need to build a function that saves a bitmap as a png.
#keep track of stimcode, shape, xtess, ttess
#

def make_movie(shape, stimulus, movie_save_name, movie_save_path, duration = 3, x = 360, height = 10, randomize_x = False, randomize_t = False, edge_speed = 240, edge_width = 30, fps = 30, xt_plot_filepath = None):
    """
    make stimulus an np array that can be passed into get_bitmap or the string "interleave"
    """
    plt.rcParams['font.size'] = '16'

    fig = plt.figure(figsize = (10, 4))

    ax_dict = fig.subplot_mosaic(
        """
        AA.BBBB
        AA.BBBB
        AA.BBBB
        CC.BBBB
        """
    )

    ax = ax_dict['A']
    plt.sca(ax)
    plt.axis('off')
    if xt_plot_filepath:
        im = image.imread(xt_plot_filepath)
        
        plt.imshow(im)

    ax = ax_dict['C']
    plt.sca(ax)
    plt.axis('off')

    ax = ax_dict['B']
    plt.sca(ax)


    total_time_steps = duration * fps

    xtess = int(360/shape[1]/edge_width)
    ttess = int(duration / (edge_width/edge_speed * shape[0]))

    if stimulus == "interleave":
        bitmap = np.array([[0.5]])
    else:
        bitmap = get_bitmap(stimulus, shape, xtess, ttess, x = x, t = total_time_steps, output_range = (0, 1))

    #plt.imshow(bitmap, aspect = 'auto', cmap = 'grey', extent = (-0, 360, -0, height))

    if randomize_x:
        x_phase = (np.random.rand(height) * np.shape(bitmap)[1]).astype(int)
    else:
        x_phase = (np.zeros(height) + np.random.rand() * np.shape(bitmap)[1]).astype(int)
    if randomize_t:
        t_phase = (np.random.rand(height) * np.shape(bitmap)[0]).astype(int)
    else:
        t_phase = (np.zeros(height) + np.random.rand() * np.shape(bitmap)[0]).astype(int)


    movie_frame_list = list()

    if not os.path.exists(movie_save_path + movie_save_name):
        print(f"making path {movie_save_path + movie_save_name}")
        os.mkdir(movie_save_path + movie_save_name)
    
    print("Making movie " + movie_save_name)

    for step in range(0, total_time_steps):
        ax.clear()
        
        plt.xlabel("Azimuth (°)")
        ax.set_xticks([0, 120, 240, 360])

        movie_frame = np.zeros((height, np.shape(bitmap)[1]))
        for row_num in range(0, height):
            movie_frame[row_num, :] = np.roll(bitmap[(step + t_phase[row_num])%np.shape(bitmap)[0]], x_phase[row_num])

        plt.imshow(movie_frame, aspect = 'auto', cmap = 'Greys', interpolation = 'none', extent = (-0, 360, -0, height), vmin = 0, vmax = 1)

        plt.yticks([])



        fig.savefig(movie_save_path+movie_save_name + f'/frame_{step}.png', bbox_inches='tight', dpi=300)
        movie_frame= imageio.imread(movie_save_path+movie_save_name + f'/frame_{step}.png')
        movie_frame_list.append(movie_frame)

        if step % fps == 0:
            print (f'frame {step} of {total_time_steps} done')
    
    imageio.mimsave(movie_save_path+ movie_save_name +'.mp4', movie_frame_list, fps=fps)

    #plt.close()


def bitmap_as_png(stimulus, shape, xtess, ttess, x, t, edge_speed = -1, priority = "xtess", path= (os.path.dirname(__file__) + "/me_stimuli_images")):
    '''
    Saves a bitmap as a png image. Uses get_bitmap
    Bitmap named as x + "x" + t + ", xtess=" + xtess + ", ttess=" + ttess + ", shape=" + shape + ", " + stimulusAsInteger
    See get_bitmap for further documentation
    '''

    bitmap = get_bitmap(stimulus, shape, xtess, ttess, x, t, (0, 1), edge_speed, priority)

    stimulus = np.int8(np.equal(stimulus, 1))
    result = int("".join(str(i) for i in stimulus),2)
    
    result = stimulus_to_int(stimulus)

    filename = str(x) + "x" + str(t) + ",xtess=" + str(xtess) + ",ttess=" + str(ttess) + ",shape=(" + str(shape[0]) + "," + str(shape[1]) + ")," + str(result)
    Image.fromarray(np.uint8(bitmap * 255), 'L').save(path + "/" + filename + ".png")

    
def stimulus_to_int(stimulus):
    stimulus = np.int8(np.equal(stimulus, 1))
    result = int("".join(str(i) for i in stimulus),2)
    return result


#Question - do we want possible bitmap stimuli to be constrained to n \in \mathbb{Z} full periods?
#Should investigate the impact of discontinuitites on motion prediction with the models
#For edge speed, assumes that the entire stimulus is 360 degrees and half a second. This is necessary for when plugging it into the model.
def get_bitmap(stimulus, shape,  xtess = 2, ttess = 2, x = 720, t = 500, output_range = (-2, 2), edge_speed = -1, priority = "xtess") -> np.ndarray:
    '''
    Gets the stimulus in the format of a 2D bitmap. 

    Parameters
    ----------
    stimulus : np array
        the stimulus in a vector form. Every pair of dimensions in the vector represents one cell in the stimulus, 
        with the first dimension of the pair representing contrast (1 for light and -1 for dark) and the second dimension of the pair
        representing the direction (1 for right and -1 for left).
        Cells are listed from right to left in x, and then from top to bottom in t.
        Example: The vector [1, 1, 1, -1, -1, 1, -1, -1] is decoded by separating the vector into pairs of dimensions.
            The first pair [1, 1] codes for a light, rightwards edge.
            The second pair [1, -1] codes for a light, leftwards edge.
            The 3rd pair [-1, 1] codes for a dark, rightwards edge.
            The 4th pair [-1, -1] codes for a dark, leftwards edge.
        The arrangement of these unit cells is dictated by the shape parameter.
    shape: tuple
        the shape of the stimulus - (rows, cols). the rows multiplied by the columns should equal half of the length of the stimulus vector.
    xtess : int
        number of times the stimulus appears in the x dimension
    ttess : int
        number of times the stimulus appears in the t dimension
    x : int
        the pixels along x. Defaults to 720 (each pixel represents 0.5 degrees).
    t : int 
        the pixels along t. Defaults to 500 ms.
    output_range: tuple
        the range of output values. Defaults to (-2, 2).
    edge_speed: int
        Optional. The speed of a single moving edge, in degrees per second - an optional argument. Overrides either xtess or ttess (see priority)
    priority: string
        Either "xtess" or "ttess". If edge_speed is specified, prioritize either xtess or ttess to create stimuli with moving edges of the desired speed.

    Returns
    -------
    bitmap : np array
        a 2D array representing the bitmap.
    '''
    cols = shape[1]
    rows = shape[0]
    #bitmap = np.zeros((t, x))

    expanded_xsize = (int)(np.lcm(x, xtess*cols))
    expanded_tsize = (int)(np.lcm(t, ttess*rows))
    xstep = (int)(expanded_xsize/xtess)
    tstep = (int)(expanded_tsize/ttess)
    
    if edge_speed != -1:
        if priority == "xtess":
            #expanded_xsize = np.lcm(x, xtess*cols)
            unit_tsize = (int)(x*500*rows/(np.gcd(x*500*rows, xtess * cols * edge_speed)))
            ttess = ((int)((int)(t*xtess*cols*edge_speed/np.gcd(x*500*rows, xtess*cols*edge_speed))/unit_tsize) + 1)
            expanded_tsize = ttess * unit_tsize


        elif priority == "ttess":
            #expanded_tsize = np.lcm(t, ttess*rows)
            unit_xsize = (int)(t*edge_speed*cols/(np.gcd(t*edge_speed*cols, ttess*rows*500)))
            xtess = ((int)((int)(x*ttess*rows*500/np.gcd(t*edge_speed*cols, ttess*rows*500))/unit_xsize) + 1)
            expanded_xsize = xtess * unit_xsize

        else:
            print("ERROR: edge speed provided, but xtess or ttess priority not specified")
            return

    bitmap = np.zeros((expanded_tsize, expanded_xsize))

    xstep = (int)(expanded_xsize/xtess)
    tstep = (int)(expanded_tsize/ttess)

    for xiter in range(xtess):
        for titer in range(ttess):
            bitmap[titer*tstep:(titer + 1)*tstep, xiter*xstep:(xiter+1)*xstep] = __bitmap_cells(stimulus, shape, xstep, tstep)

    center = (output_range[0] + output_range[1])/2
    spread = output_range[1]-output_range[0]

    bitmap *= spread
    bitmap = bitmap + center - (0.5 * spread)

    if edge_speed != -1:
        if priority == "xtess":
            bitmap = bitmap[:(int)(t*xtess*cols*edge_speed/np.gcd(x*500*rows, xtess*cols*edge_speed)),:]
        elif priority == "ttess":
            bitmap = bitmap[:,:(int)(x*ttess*rows*500/np.gcd(t*edge_speed*cols, ttess*rows*500))]

    #bitmap = resize(bitmap, (t, x))
    bitmap = downscale_local_mean(bitmap, ((int)(np.shape(bitmap)[0]/t), (int)(np.shape(bitmap)[1]/x)))

    return bitmap

def __bitmap_cells(stimulus, shape, x, t) -> np.ndarray:
    '''
    Called by get_bitmap to get a bitmap for each individual cell from a given stimulus.

    Parameters
    ----------
    stimulus : np array
        the stimulus in the form of a vector.
    shape: tuple
        the shape of the stimulus - (rows, cols)
    x : int
        the number of x pixels afforded to the cell.
    t : int
        the number of t pixels afforded to the cell.

    Returns
    -------
    bitmap: np array
    a 2D array representing the bitmap. 
    
    '''
    xstep = (int)(x/shape[1])
    tstep = (int)(t/shape[0])
    
    startindex = 0

    bitmap = np.zeros((t, x))
    tvals, xvals = np.meshgrid(np.arange(0, tstep),np.arange(0, xstep), indexing = 'ij')
    tvals = (tvals*xstep + xstep/2)
    xvals = xvals*tstep + tstep/2

    for rownum in range(shape[0]):
        for colnum in range(shape[1]):

            cell = stimulus[startindex:startindex +2]

            if np.array_equal(cell, np.array([1, 1])):
                bitmap[rownum*tstep:(rownum + 1) * tstep, colnum*xstep: (colnum + 1)*xstep] = (np.less_equal(xvals, tvals).astype(np.float32) + np.less(xvals, tvals).astype(np.float32))/2 #light rightwards
            elif np.array_equal(cell, np.array([1, -1])):
                bitmap[rownum*tstep:(rownum + 1 )* tstep, colnum*xstep: (colnum + 1)*xstep] = (np.less_equal(tstep*xstep-xvals, tvals).astype(np.float32) + np.less(tstep*xstep-xvals, tvals).astype(np.float32))/2 #light leftwards
            elif np.array_equal(cell, np.array([-1, 1])):
                bitmap[rownum*tstep:(rownum + 1 )* tstep, colnum*xstep: (colnum + 1)*xstep] = (np.greater_equal(xvals, tvals).astype(np.float32) + np.greater(xvals, tvals).astype(np.float32))/2 #dark rightwards  
            else:
                bitmap[rownum*tstep:(rownum + 1 )* tstep, colnum*xstep: (colnum + 1)*xstep] = (np.greater_equal(tstep*xstep-xvals, tvals).astype(np.float32) + np.greater(tstep*xstep-xvals, tvals).astype(np.float32))/2 #dark leftwards

            startindex += 2
    
    return bitmap

def display_stimulus(stimulus, shape, title, xtess = 2, ttess = 2, flip_axes = False) -> None:
    '''
    Uses matplotlib to create a visualization of the stimulus.

    Parameters
    ----------
    stimulus : np array
        the stimulus in the form of a vector.
    shape: tuple
        the shape of the stimulus - (rows, cols)
    title : string
        the title of the display.
    xtess : int
        number of times the stimulus appears in the x dimension
    ttess : int
        number of times the stimulus appears in the t dimension

    Returns
    -------
    nothing  
    '''
    
    cols = shape[1]
    rows = shape[0]

    for xiter in range(xtess):
            for yiter in range(ttess):
                __print_cells(stimulus, shape, xiter * cols, yiter * rows, flip_axes)

    plt.title(title)

    if flip_axes:
        plt.xlabel('Time')
        plt.ylabel('Space')
        ax = plt.gca()

        ax.set_xlim(0, shape[0] * ttess)
        ax.set_ylim(0, shape[1] * xtess)
        #ax.xaxis.set_label_position('top')
        #ax.xaxis.tick_top()
        #plt.gca().invert_yaxis()
    else:
        plt.ylabel('Time')
        plt.xlabel('Space')
        ax = plt.gca()
        #ax.xaxis.set_label_position('top')
        #ax.xaxis.tick_top()

        ax.set_ylim(0, shape[0] * ttess)
        ax.set_xlim(0, shape[1] * xtess)

        ax.invert_yaxis()


def __print_cells(stimulus, shape, xpos, ypos, flip_axes) -> None:
    '''
    Called by display_stimulus to print out each individual cell from a given stimulus.

    Parameters
    ----------
    stimulus : np array
        the stimulus in the form of a vector.
    shape: tuple
        the shape of the stimulus - (rows, cols)
    xpos : int
        the starting x position.
    ypos : int
        the starting y position.

    Returns
    -------
    nothing 
    
    '''
    startindex = 0
    if flip_axes:
        for rownum in range(shape[0]):
            for colnum in range(shape[1]):

                cell = stimulus[startindex:startindex +2]

                if np.array_equal(cell, np.array([1, 1])):
                    plt.fill([0 + ypos+rownum, 0 + ypos+rownum, 1 + ypos+rownum], [0 + xpos+colnum, 1 + xpos+colnum, 1 + xpos+colnum], 'black') #light rightwards
                elif np.array_equal(cell, np.array([1, -1])):
                    plt.fill([0 + ypos+rownum, 1 + ypos+rownum, 0 + ypos+rownum], [0 + xpos+colnum, 0 + xpos+colnum, 1 + xpos+colnum], 'black') #light leftwards
                elif np.array_equal(cell, np.array([-1, 1])):
                    plt.fill([0 + ypos+rownum, 1 + ypos+rownum, 1 + ypos+rownum], [0 + xpos+colnum, 0 + xpos+colnum, 1 + xpos+colnum], 'black') #dark rightwards  
                else:
                    plt.fill([1 + ypos+rownum, 0 + ypos+rownum, 1 + ypos+rownum], [0+xpos+colnum, 1+xpos+colnum, 1+xpos+colnum], 'black') #dark leftwards

                startindex += 2
    else:
        for rownum in range(shape[0]):
            for colnum in range(shape[1]):

                cell = stimulus[startindex:startindex +2]

                if np.array_equal(cell, np.array([1, 1])):
                    plt.fill([0 + xpos+colnum, 1 + xpos+colnum, 1 + xpos+colnum], [0 + ypos+rownum, 0 + ypos+rownum, 1 + ypos+rownum], 'black') #light rightwards
                elif np.array_equal(cell, np.array([1, -1])):
                    plt.fill([0 + xpos+colnum, 0 + xpos+colnum, 1 + xpos+colnum], [0 + ypos+rownum, 1 + ypos+rownum, 0 + ypos+rownum], 'black') #light leftwards
                elif np.array_equal(cell, np.array([-1, 1])):
                    plt.fill([0 + xpos+colnum, 0 + xpos+colnum, 1 + xpos+colnum], [0 + ypos+rownum, 1 + ypos+rownum, 1 + ypos+rownum], 'black') #dark rightwards  
                else:
                    plt.fill([0+xpos+colnum, 1+xpos+colnum, 1+xpos+colnum], [1 + ypos+rownum, 0 + ypos+rownum, 1 + ypos+rownum], 'black') #dark leftwards

                startindex += 2
    


def get_c_flip(shape) -> np.ndarray:
    '''
    Gets a matrix that can be multiplied to a stimulus vector to flip contrast.

    Parameters
    ----------
    shape: tuple
        the shape of the stimulus - (rows, cols)

    Returns
    -------
    c_flip : np array
        a length x length matrix that flips contrast when multiplied to a stimulus vector.
    '''
    cells = shape[0]*shape[1]
    length = cells*2
    c_flip = np.zeros((length, length))
    for i in range(length):
        c_flip[i, i] = (i % 2) * 2 - 1

    return c_flip

def get_x_flip(shape)-> np.ndarray:
    '''
    Gets a matrix that can be multiplied to a stimulus vector to flip x.

    Parameters
    ----------
    shape: tuple
        the shape of the stimulus - (rows, cols)

    Returns
    -------
    x_flip : np array
        a length x length matrix that flips x when multiplied to a stimulus vector.
    '''
    cells = shape[0]*shape[1]

    cols = shape[1]
    length = cells*2
    sublength = cols*2
    x_flip = np.zeros((length, length))
    subunit = np.zeros((sublength, sublength))
    for i in range(sublength):
        subunit[i, sublength + 2*(i%2) - 2 - i] = (i % 2) * -2 + 1
    for i in range(shape[0]):
        x_flip[i*sublength:(i*sublength+sublength), i*sublength:(i*sublength+sublength)] = subunit
    return x_flip

def get_t_flip(shape)-> np.ndarray:
    '''
    Gets a matrix that can be multiplied to a stimulus vector to flip time.

    Parameters
    ----------
    shape: tuple
        the shape of the stimulus - (rows, cols)

    Returns
    -------
    t_flip : np array
        a length x length matrix that flips time when multiplied to a stimulus vector.
    '''
    cells = shape[0]*shape[1]
    cols = shape[1]

    length = cells*2
    sublength = cols*2
    subunit = np.zeros((sublength, sublength))
    t_flip = np.zeros((length, length))

    for i in range(sublength):
        subunit[i, i] = -1
    for i in range(shape[0]):
        t_flip[i*sublength:(i*sublength + sublength), (length - sublength - i*sublength):(length - sublength - (i*sublength)+sublength)] = subunit
        
    return t_flip

def get_shift_list(shape)-> list:
    '''
    Gets a set of matrices that can be multiplied to a stimulus vector to represent translations in x and y.

    Parameters
    ----------
    shape: tuple
        the shape of the stimulus - (rows, cols)

    Returns
    -------
    shift_list : list
        a list of (length x length) matrices that represents translations across x and y.
    '''
    cells = shape[0]*shape[1]
    length = cells*2
    cols = shape[1]
    sublength = cols*2
    x_shift = np.zeros((length, length))
    subunit = np.zeros((sublength, sublength))
    for i in range(sublength):
        subunit[i, i-2] = 1
    
    for i in range(shape[0]):
        x_shift[i*sublength:(i*sublength+sublength), i*sublength:(i*sublength+sublength)] = subunit

    x_shift_list = list()
    for i in range(cols):
        x_shift_list.append(np.matmul(np.identity(length), np.linalg.matrix_power(x_shift, i)))
    
    y_shift = np.zeros((length, length))
    for i in range(length):
        y_shift[i, i-(sublength)]=1

    y_shift_list = list()
    for i in range(shape[0]):
        y_shift_list.append(np.matmul(np.identity(length), np.linalg.matrix_power(y_shift, i)))

    shift_list = list()
    for x_shift in x_shift_list:
        for y_shift in y_shift_list:
            shift_list.append(np.matmul(x_shift, y_shift))

    return shift_list


def __directions_only(shape)-> np.ndarray:
    '''
    Gets a matrix that can be multiplied to a stimulus vector to only look at directions. 
    Parameters
    ----------
    shape: tuple
        the shape of the stimulus - (rows, cols)

    Returns
    -------
    direction_matrix : np array
        a length x length matrix that zeroes contrast but preserves direction.
    '''
    cells = shape[0]*shape[1]
    length = cells*2
    direction_matrix = np.zeros((length, length))
    for i in range(1, length, 2):
        direction_matrix[i, i] = 1
    return direction_matrix

def __polarity_only(shape)-> np.ndarray:
    '''
    Gets a matrix that can be multiplied to a stimulus vector to only look at polarity. 
    Parameters
    ----------
    shape: tuple
        the shape of the stimulus - (rows, cols)

    Returns
    -------
    direction_matrix : np array
        a length x length matrix that zeroes contrast but preserves direction.
    '''
    cells = shape[0]*shape[1]
    length = cells*2
    direction_matrix = np.zeros((length, length))
    for i in range(0, length, 2):
        direction_matrix[i, i] = 1
    return direction_matrix

def __get_discontinuity_fraction(stimulus, shape, polarity_matrix = None) -> float:
    if not (type(polarity_matrix) is np.ndarray):
        polarity_matrix =__polarity_only(shape)
    cells = shape[0]*shape[1]
    pol = np.matmul(polarity_matrix, stimulus)
    rolled_pol = np.roll(pol, shape[1]*2)
    return np.sum((pol * rolled_pol) == 1)/cells


def transform_stimulus(stimulus, shape, flips) -> np.ndarray:
    '''
    flips along x, c, and t in varying combos.

    Parameters
    ----------
    stimulus : np array
        the stimulus in a vector form
    shape: tuple
        the shape of the stimulus - (rows, cols)
    flips: string
        the flips to perform

    Returns
    -------
    result : np array
    a transformed stimulus according to flips
    '''

    c_flip = get_c_flip(shape)
    x_flip = get_x_flip(shape)
    t_flip = get_t_flip(shape)

    for i in flips:
        if(i == 'x'):
            stimulus = np.matmul(x_flip, stimulus)
        elif(i == 'c'):
            stimulus = np.matmul(c_flip, stimulus)
        elif(i == 't'):
            stimulus = np.matmul(t_flip, stimulus)
    
    return stimulus


def check_symmetry(stimulus, shape, symmetry_type = ['t', 'c', 'x', 'tc', 'tx', 'cx', 'xtc'], shift_list = None)-> list:
    '''
    Evaluates whether a stimulus exhibits a symmetry of a given type.

    Parameters
    ----------
    stimulus : np array
        the stimulus in a vector form
    shape: tuple
        the shape of the stimulus - (rows, cols)
    symmetry_type: string or list
        The symmetry(s) to test

    Returns
    -------
    is_symmetric : boolean array
    whether or not the stimulus is exhibits symmetry of type symmetry_type
    '''
    if type(symmetry_type) is list:
        is_symmetric = []
        for i in symmetry_type:
            is_symmetric.append(check_symmetry(stimulus, shape, i))
        return is_symmetric

    transformed_stimulus = transform_stimulus(stimulus, shape, symmetry_type)

    if not(type (shift_list) is list):
        print("oops")
        shift_list = get_shift_list(shape)

    for matrix in shift_list:
        if np.array_equal(np.matmul(matrix, transformed_stimulus), stimulus):
            return True
    return False

def check_disparity_from_symmetry(stimulus, shape, symmetry_type = ['t', 'c', 'x', 'tc', 'tx', 'cx', 'xtc'], shift_list = None) -> list:
    '''
     Evaluates how far a stimulus is from exhibiting a symmetry of a given type.

    Parameters
    ----------
    stimulus : np array
        the stimulus in a vector form
    shape: tuple
        the shape of the stimulus - (rows, cols)
    symmetry_type: string or list
        The symmetry(s) to test

    Returns
    -------
    disparity : boolean array
    the smallest fraction of cells that are different from yielding a given symmetry
    '''
    if type(symmetry_type) is list:
        disparity = []
        for i in symmetry_type:
            disparity.append(check_disparity_from_symmetry(stimulus, shape, i))
        return disparity
    
    transformed_stimulus = transform_stimulus(stimulus, shape, symmetry_type)

    if not(type (shift_list) is list):
        print("oops")
        shift_list = get_shift_list(shape)

    
    num_cells = int(np.size(stimulus)/2)
    disparity_per_shift = np.zeros(np.size(shift_list, 0))

    for a, matrix in enumerate(shift_list, 0):
        different_cells = 0
        shifted_stimulus = np.matmul(matrix, stimulus)
        for i in range(0, num_cells):
            if(not np.array_equal(transformed_stimulus[2*i:2*i+2], shifted_stimulus[2*i:2*i + 2])):
                different_cells = different_cells + 1
        
        disparity_per_shift[a] = different_cells/num_cells

            
    return disparity_per_shift.min()
        
def fast_symmetry_table(shape, symmetries = ['t', 'c', 'x', 'tc', 'tx', 'cx', 'xtc'], mode = 'symmetry')-> list:
    '''
    Creates a table of symmetries.

    Parameters
    ----------
    shape: tuple
        the shape of the stimulus - (rows, cols)
    symmetries: list
        a list of all the symmetries to evaluate
    mode: string
        Either 'symmetry' or 'disparity'. Symmetry just returns true/false, disparity returns the fraction of cells that are different,
        preventing it from being of a given symmetry

    Returns
    -------
    table : list
        a list of lists representing the table.
    '''
    if(mode != 'symmetry' and mode != 'disparity'):
        print('CHOOSE EITHER SYMMETRY OR DISPARITY AS THE MODE')
        return

    cells = shape[0]*shape[1]
    table = [['stimulus', 'net movement', 'discontinuity fraction'] + symmetries]

    direction_matrix = __directions_only(shape)
    polarity_matrix =__polarity_only(shape)
    shift_list = get_shift_list(shape)

    #we only care about the first quarter (that start with 1, 1)
    #is there a way to only generate balanced things? We know the first two terms must be 0. Out of 32 places, 16 should be 1s. 30 choose 16
    #use itertools

    count = 0
    start_time = time.time()
    total = math.comb(cells * 2 - 2, cells)

    for negative_placements in itertools.combinations(range(2, cells * 2), cells):
        count+=1
        if count%(2**(21)) == 0:
            print((str(count) + ' done out of ' + str(total) + ' (' + str(count/total) + ')'))
            end_time = time.time()
            print('split time = ' + str(end_time - start_time))
            start_time = time.time()


        stimulus = np.ones(32).astype('int32')
        for index in negative_placements:
            stimulus[index] = -1

        net_direction =np.sum((np.matmul(direction_matrix, stimulus)))
        
        #if statement is for fast running
        if net_direction != 0:
            continue
        
        discontinuity_fraction = __get_discontinuity_fraction(stimulus, shape, polarity_matrix)

        #if statement is for fast running
        if discontinuity_fraction != 0:
            continue
            

        next_row = [list(stimulus), net_direction, discontinuity_fraction]

        #check t symmetry first. 
        if mode == 'symmetry':
            t_value = check_symmetry(stimulus, shape, table[0][3], shift_list)
        else:
            t_value = check_disparity_from_symmetry(stimulus, shape, table[0][3], shift_list)

        if (mode == 'symmetry' and t_value == False) or (mode != 'symmetry' and t_value != 0):
            continue

        next_row.append(t_value)

        for i in range(4, len(table[0])):
            if mode == 'symmetry':
                next_row.append(check_symmetry(stimulus, shape, table[0][i]), shift_list)
            else:
                next_row.append(check_disparity_from_symmetry(stimulus, shape, table[0][i], shift_list))
        table.append(next_row)

    return table




def symmetry_table(shape, symmetries = ['t', 'c', 'x', 'tc', 'tx', 'cx', 'xtc'], mode = 'symmetry')-> list:
    '''
    Creates a table of symmetries.

    Parameters
    ----------
    shape: tuple
        the shape of the stimulus - (rows, cols)
    symmetries: list
        a list of all the symmetries to evaluate
    mode: string
        Either 'symmetry' or 'disparity'. Symmetry just returns true/false, disparity returns the fraction of cells that are different,
        preventing it from being of a given symmetry

    Returns
    -------
    table : list
        a list of lists representing the table.
    '''
    if(mode != 'symmetry' and mode != 'disparity'):
        print('CHOOSE EITHER SYMMETRY OR DISPARITY AS THE MODE')
        return

    cells = shape[0]*shape[1]
    table = [['stimulus', 'net movement', 'discontinuity fraction'] + symmetries]

    direction_matrix = __directions_only(shape)
    polarity_matrix =__polarity_only(shape)
    shift_list = get_shift_list(shape)

    #we only care about the first quarter (that start with 1, 1)
    #is there a way to only generate balanced things? We know the first two terms must be 0. Out of 32 places, 16 should be 1s. 30 choose 16
    for i in range(2**(cells*2)):

        #loop through using binary numbers
        binary = np.binary_repr(i, width = cells*2)
        stimulus_as_list = list(binary)
        for i in range(len(stimulus_as_list)):
            if(stimulus_as_list[i] == '0'):
                stimulus_as_list[i] = 1
            else:
                stimulus_as_list[i] = -1

        stimulus = np.array(stimulus_as_list)


        net_direction =np.sum((np.matmul(direction_matrix, stimulus)))
        
        
        discontinuity_fraction = __get_discontinuity_fraction(stimulus, shape, polarity_matrix)

            

        next_row = [stimulus_as_list, net_direction, discontinuity_fraction]


        for i in range(3, len(table[0])):
            if mode == 'symmetry':
                next_row.append(check_symmetry(stimulus, shape, table[0][i]), shift_list)
            else:
                next_row.append(check_disparity_from_symmetry(stimulus, shape, table[0][i], shift_list))
        table.append(next_row)

    return table


if __name__ =="__main__":
    print('working!')

    print(check_symmetry([1, 1, 1, -1, 1, 1, 1, -1] , (2, 2)))
    print(check_disparity_from_symmetry([1, 1, 1, -1, 1, 1, 1, -1] , (2, 2)))
    #use this area for debugging.

#Tuning curves - take in: type of stimulus (sine wave, square wave), frequency, velocity
 #sinusoids at different velocities
    #sawtooth
    #my stimuli
    #single moving edge
    #Build a harness to validate a model - sine wave gratings at different velocities
    #build a 30 degree sine wave grating - present a full period in time
    #
#and model parameters


    

