#!/usr/bin/python

import os, sys
import json
import numpy as np
import re

### YOUR CODE HERE: write at least three functions which solve
### specific tasks by transforming the input x and returning the
### result. Name them according to the task ID as in the three
### examples below. Delete the three examples. The tasks you choose
### must be in the data/training directory, not data/evaluation.


################################################################################################################################
def solve_007bbfb7(x):
    """
    Task Description:
        Input:
            [[7 0 7]
            [7 0 7]
            [7 7 0]]
        Output:
            [[7 0 7 0 0 0 7 0 7]
            [7 0 7 0 0 0 7 0 7]
            [7 7 0 0 0 0 7 7 0]
            [7 0 7 0 0 0 7 0 7]
            [7 0 7 0 0 0 7 0 7]
            [7 7 0 0 0 0 7 7 0]
            [7 0 7 7 0 7 0 0 0]
            [7 0 7 7 0 7 0 0 0]
            [7 7 0 7 7 0 0 0 0]]
    
    Colour Encoding:
        Black = 0, Dark Blue = 1, Red = 2 , Green = 3 , Yellow = 4 , Grey = 5 , Pink = 6 , Orange = 7 , Sky Blue = 8 , Brown = 9
        
    Algorithm:
        The output matrix's dimension is square of input matrix's dimension. The output matrix gets a copy of full input matrix
        if the cell in input matrix in non-black (0). For efficient copy we have used numpy slicing technique.
        
    Results:
        All the 5 train test cases and 1 testing test cases passed
    """
    assert type(x) == np.ndarray
    height,width = x.shape                                                    # height and width of numpy 2D array
    x_answer = np.zeros((height*height,width*width))                          # output matrix dimension square of input dimensions
    # Reference :  https://stackoverflow.com/questions/44092848/get-indicies-of-non-zero-elements-of-2d-array
    nonzero = np.nonzero(x)                                                   # Returns a tuple of (nonzero_row_index, nonzero_col_index)
    nonzero_row = nonzero[0]
    nonzero_col = nonzero[1]

    for row, col in zip(nonzero_row, nonzero_col):                            # for each non zero cell we will copy original matrix in new output matrixS
       x_answer[height*row:height*row+height,width*col:width*col+width] = x   # limits of indexes,height and width

    return x_answer.astype(int)           

################################################################################################################################
    
from collections import defaultdict

def isOutlier(x,i,j):
    # This function finds number of neighbour of a cell and counts of unique colour of neighbour which are not as same colour as the cell
    height,width = x.shape
    D = defaultdict(int)
    num_neighbour = 0
    
    if i>0:
        num_neighbour +=1
        if x[i-1][j]!=x[i][j]:
            D[x[i-1][j]] +=1
            
    if i>0 and j>0:
        num_neighbour +=1
        if x[i-1][j-1]!=x[i][j]:
            D[x[i-1][j-1]] +=1
        
    if i>0 and j+1<width:
        num_neighbour +=1
        if x[i-1][j+1]!=x[i][j]:
            D[x[i-1][j+1]] +=1
        
    if j+1<width:
        num_neighbour +=1
        if x[i][j+1]!=x[i][j]:
            D[x[i][j+1]] +=1
        
    if i+1<height and j+1<width:
        num_neighbour +=1
        if x[i+1][j+1]!=x[i][j]:
            D[x[i+1][j+1]] +=1
        
    if i+1<height:
        num_neighbour +=1
        if x[i+1][j]!=x[i][j]:
            D[x[i+1][j]] +=1
        
    if i+1<height and j>0:
        num_neighbour +=1
        if x[i+1][j-1]!=x[i][j]:
            D[x[i+1][j-1]] +=1
        
    if j>0:
        num_neighbour +=1
        if x[i][j-1]!=x[i][j]:
            D[x[i][j-1]] +=1
       
    if sum(D.values()) == num_neighbour: # if all the neighbours have different colour that the cell then its a outlier
        inverse = [(value, key) for key, value in D.items()]
        return max(inverse)[1] # we will return the mapping colour as the colour which occured most time, solved on paper this works well
    return -1
 
################################################################################################################################    
    
def solve_d07ae81c(x):
    """
    Task Description:
        Input:
            [[8 8 8 3 3 3 3 3 3 8 4 8 8 8 8 8 8 8 8]
            [8 8 8 3 3 3 3 3 3 8 8 8 8 8 8 8 8 8 8]
            [8 8 8 3 3 3 3 3 3 8 8 8 8 8 8 8 8 8 8]
            [8 8 8 3 3 3 3 3 3 8 8 8 8 8 8 8 8 8 8]
            [3 3 3 3 3 3 1 3 3 3 3 3 3 3 3 3 3 3 3]
            [3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3]
            [3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3]
            [8 8 8 3 3 3 3 3 3 8 8 8 8 8 8 8 8 8 8]
            [8 8 8 3 3 3 3 3 3 8 8 8 8 8 8 8 8 8 8]
            [8 8 8 3 3 3 3 3 3 8 8 8 8 8 8 8 8 8 8]
            [8 8 8 3 3 3 3 3 3 8 8 8 8 8 8 8 8 8 8]
            [3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3]
            [3 3 3 3 3 3 3 3 3 3 3 3 3 3 1 3 3 3 3]
            [3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3]
            [3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3]
            [8 8 8 3 3 3 3 3 3 8 8 8 8 8 8 8 8 8 8]
            [8 8 8 3 3 3 3 3 3 8 8 8 8 8 8 8 8 8 8]]
        Output:
            [[8 8 4 3 3 3 3 3 3 8 4 8 8 8 8 8 8 8 8]
            [8 8 8 1 3 3 3 3 3 4 8 4 8 8 8 8 8 8 8]
            [8 8 8 3 1 3 3 3 1 8 8 8 4 8 8 8 8 8 8]
            [8 8 8 3 3 1 3 1 3 8 8 8 8 4 8 8 8 8 8]
            [3 3 3 3 3 3 1 3 3 3 3 3 3 3 1 3 3 3 3]
            [3 3 3 3 3 1 3 1 3 3 3 3 3 3 3 1 3 3 3]
            [3 3 3 3 1 3 3 3 1 3 3 3 3 3 3 3 1 3 3]
            [8 8 8 1 3 3 3 3 3 4 8 8 8 8 8 8 8 4 8]
            [8 8 4 3 3 3 3 3 3 8 4 8 8 8 8 8 8 8 4]
            [8 4 8 3 3 3 3 3 3 8 8 4 8 8 8 8 8 4 8]
            [4 8 8 3 3 3 3 3 3 8 8 8 4 8 8 8 4 8 8]
            [3 3 3 3 3 3 3 3 3 3 3 3 3 1 3 1 3 3 3]
            [3 3 3 3 3 3 3 3 3 3 3 3 3 3 1 3 3 3 3]
            [3 3 3 3 3 3 3 3 3 3 3 3 3 1 3 1 3 3 3]
            [3 3 3 3 3 3 3 3 3 3 3 3 1 3 3 3 1 3 3]
            [8 8 8 3 3 3 3 3 3 8 8 4 8 8 8 8 8 4 8]
            [8 8 8 3 3 3 3 3 3 8 4 8 8 8 8 8 8 8 4]]
    
    Colour Encoding:
        Black = 0, Dark Blue = 1, Red = 2 , Green = 3 , Yellow = 4 , Grey = 5 , Pink = 6 , Orange = 7 , Sky Blue = 8 , Brown = 9
        
    Algorithm:
        We are provided with a 2D numpy matrix with has 2 backgrouds and some outliers , the output matrix we need to change colour
        of all cells which are in diagonal of those outliers. We first need to find outliers (used isOutlier function for this) and
        save the colour mapping which will be used to change the colour of cell later. The isOutlier function returns the colour which
        will be changed to colour of outlier later.
        
    Results:
        All the 3 train test cases and 1 testing test cases passed
    """
    height,width = x.shape                  # height and width of numpy 2D array    
    fillmap = dict()                        # dictinary to store input colour to output colour for later diagonal operations
    for i in range(height):
        for j in range(width):
            color = isOutlier(x,i,j)        # check if cell i,j is outlier or not 
            if color !=-1:                  # if isOutlier returns -1 , otherwise a not a outlier and returns a colour
                fillmap[color] = x[i][j]    # keep colour mapping saved

    x_answer = x.copy()
    for i in range(height):
        for j in range(width):
            color = isOutlier(x,i,j)        # check if cell i,j is outlier or not 
            if color !=-1:                  # if the cell is outlier change colour of all the diagonal cell in all 4 possible directions
                k,m=i-1,j-1
                while k>=0 and m>=0:
                    if x[k][m] in fillmap:
                        x_answer[k][m] = fillmap[x[k][m]] # change colour using stored colour mapping
                    k-=1;m-=1
                    
                k,m=i+1,j+1
                while k<height and m<width:
                    if x[k][m] in fillmap:
                        x_answer[k][m] = fillmap[x[k][m]] # change colour using stored colour mapping
                    k+=1;m+=1
                    
                k,m=i+1,j-1
                while k<height and m>=0:
                    if x[k][m] in fillmap:
                        x_answer[k][m] = fillmap[x[k][m]] # change colour using stored colour mapping
                    k+=1;m-=1
                    
                k,m=i-1,j+1
                while k>=0 and m<width:
                    if x[k][m] in fillmap:
                        x_answer[k][m] = fillmap[x[k][m]] # change colour using stored colour mapping
                    k-=1;m+=1             
    return x_answer                                            # convert matrix output to integer to match with test cases

################################################################################################################################    

def solve_a65b410d(x):
    """
    Task Description:
        Input:
        [[0 0 0 0 0 0 0]
        [0 0 0 0 0 0 0]
        [0 0 0 0 0 0 0]
        [2 2 0 0 0 0 0]
        [0 0 0 0 0 0 0]
        [0 0 0 0 0 0 0]
        [0 0 0 0 0 0 0]]
        
        Desired Output:
        [[3 3 3 3 3 0 0]
        [3 3 3 3 0 0 0]
        [3 3 3 0 0 0 0]
        [2 2 0 0 0 0 0]
        [1 0 0 0 0 0 0]
        [0 0 0 0 0 0 0]
        [0 0 0 0 0 0 0]]
    
    Colour encodings:
        Black = 0, Red = 2 , Green = 3, Blue=1 
    
    Algorithm:
        We will be finding row with colour red and fill all row above it with red with increasing counter and
        all row below it with blue colour with decreasing counter.
        
    Results:
        All the 3 train test cases and 1 testing test cases passed
    """
    assert type(x) == np.ndarray    
    x_answer = x.copy()                 # creating a copy of original matrix , our output will be of same dimension as input
    solutions = np.argwhere(x == 2)     # provides us list of coordinates with color red (2)
    index = solutions[0][0]             # x coordinate where its red (2)
    dist = len(solutions)               # the number of red tile, this will be used to fill rows above this row 
    
    # fill the above rows with green (3) increamenting the width  
    for i in range(index-1,-1,-1):      # taking reverse steps till start of matrix
        dist += 1
        x_answer[i][0:dist] = 3         # fill with green colour
        
    height = x.shape[0]
    dist = len(solutions)               # the number of red tile, this will be used to fill rows below this row
    # fill the below rows with blue (1) decrementing the width 
    for i in range(index+1,height):
        dist -= 1
        x_answer[i][0:dist] = 1         # fill with blue colour
        if dist == 0:
            break                       # if width to fill is zero break the loop
    return x_answer

################################################################################################################################

def solve_746b3537(x):
    """
    Task Description:
        Input:
            [[2 3 3 8 1]
            [2 3 3 8 1]
            [2 3 3 8 1]]
        Output:
            [[2 3 8 1]]
    
    Colour Encoding:
        Black = 0, Dark Blue = 1, Red =2 , Green = 3 , Yellow = 4 , Grey = 5 , Pink = 6 , Orange = 7 , Sky Blue = 8 , Brown = 9
        
    Algorithm:
        We will first find is it horizontal or vertically flowing matrix. After that we need to find unique colour is row or columns 
        accordingly and present as input in appropriate dimensions.
        
    Results:
        All the 4 train test cases and 1 testing test cases passed
    """
    assert type(x) == np.ndarray
    height,width = x.shape                      # height and width of numpy 2D array
    arr = x[0][0:]                              # capture the colours of first horizontal row
    isHorizantallySame = np.unique(arr).size    # capture the number of unique colours is  first horizontal row
    x_answer = []
    if isHorizantallySame == 1:                 # if there is only one unique colour its horizontal flowing matrix otherwise vertical
        # case for horizontal flowing matrix
        first = x[0][0]
        x_answer.append([first])                # append first colour in answer matrix
        for i in range(1,height):
            sec = x[i][0]
            if sec != first:                    # capture any change of colour in vertical column
                x_answer.append([sec])          # apeend that unique colour in answer matrix
                first = sec                     # update this colour for further finding of next new colour
    else:
        # case for vertical flowing matrix
        first = x[0][0]
        x_answer.append(first)                  # append first colour in answer matrix
        for i in range(1,width):
            sec = x[0][i]
            if sec != first:                    # capture any change of colour in horizontal row
                x_answer.append(sec)            # apeend that unique colour in answer matrix
                first = sec                     # update this colour for further finding of next new colour
        x_answer = [x_answer]                   # just to make visually similiar to answer 
    return np.array(x_answer)                   # return as numpy 2D array

################################################################################################################################

def solve_f2829549(x):
    """
    Task Description:
        Input:
            [[7 7 0 1 5 0 0]
            [7 0 0 1 5 0 0]
            [0 0 0 1 5 0 5]
            [0 0 0 1 5 5 0]]
        Output:
            [[0 0 3]
            [0 3 3]
            [0 3 0]
            [0 0 3]]
    
    Colour Encoding:
        Black = 0, Dark Blue = 1, Red =2 , Green = 3 , Yellow = 4 , Grey = 5 , Pink = 6 , Orange = 7 , Sky Blue = 8 , Brown = 9
        
    Algorithm:
        We will first segregate the provided matrix into 2 using the boundary created by the Dark Blue (1) color. Now consider
        the remaing parts as 2 different matrix which will be of similiar dimensions. We need to find the cells which are 
        black (0) in both matrix and create a new matrix with all those cell marked green (3) and remaining black (0).
        
    Results:
        All the 5 train test cases and 1 testing test cases passed
    """
    assert type(x) == np.ndarray
    height,width = x.shape                  # height and width of numpy 2D array
    x_left = x[:,:int(width/2)]             # the left matrix is first half of input matrix (which is on left of Dark blue boundary)
    x_right = x[:,int(width/2)+1:]          # the right matrix is second half of input matrix (which is on right of Dark blue boundary)
    x_left = np.where(x_left == 0,1,0)      # convert all cell Black (0) in left cell to 1 else 0
    x_right = np.where(x_right == 0,1,0)    # convert all cell Black (0) in right cell to 1 else 0
    x_mul = np.multiply(x_left,x_right)     # multiply both matrix , this is element wise matrix. Final matrix will have 1's in cell which were having 1 on both side matrix
    x_answer = np.where(x_mul == 1,3,0)     # convert all 1s to Green (3)
    return x_answer

################################################################################################################################

def find_different(z):   
    A = z[0]                            # store first partition
    A_copy = np.where(A == 0,1,0)       # convert all Black (0) color to 1 else 0
    unmatch_counter = 0
    index = -1
    for i in range(1,len(z)):
        B = z[i]                        # check each partition one by one
        B_copy = np.where(B == 0,1,0)   # convert all Black (0) color to 1 else 0
        if not (A_copy==B_copy).all():  # check if both matrix matches exactly element wise
            unmatch_counter +=1
            index = i                   # if not store index of that matrix
    if unmatch_counter == 1:
        return z[index]                 # the indexed matrix is unique matrix
    else:
        return z[0]                     # this case will occur if first partion was unique matrix
    
def solve_a87f7484(x):
    """
    Task Description:
        Input:
            [[6 0 6]
            [0 6 6]
            [6 0 6]
            [4 0 4]
            [0 4 4]
            [4 0 4]
            [8 8 8]
            [8 0 8]
            [8 8 8]]
        Output:
            [[8 8 8]
            [8 0 8]
            [8 8 8]]
    
    Colour Encoding:
        Black = 0, Dark Blue = 1, Red = 2 , Green = 3 , Yellow = 4 , Grey = 5 , Pink = 6 , Orange = 7 , Sky Blue = 8 , Brown = 9
        
    Algorithm:
        We will first split the given 2D numpy in 3x3 matrix and will find the matrix which is different from all other matrix.
        The split will be vertical if heigh is more than width else split will be horizontal.
        
    Results:
        All the 4 train test cases and 1 testing test cases passed
    """
    assert type(x) == np.ndarray
    height,width = x.shape              # height and width of numpy 2D array
    if height > width: 
        z1 = np.vsplit(x,height/width)  # split verticaly is height > width. create height/width number of partitions 
    else:
        z1 = np.hsplit(x, width/height) # split horizontally is width > height. create width/height number of partitions 
    return find_different(z1)           # return the unique partion using ulility function

################################################################################################################################

def solve_7468f01a(x):
    """
    Task Description:
        Input:
            [[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
            [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
            [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
            [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
            [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
            [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
            [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
            [0 0 6 6 6 3 6 6 0 0 0 0 0 0 0 0 0]
            [0 0 6 3 3 3 6 6 0 0 0 0 0 0 0 0 0]
            [0 0 6 6 6 6 3 6 0 0 0 0 0 0 0 0 0]
            [0 0 6 6 6 6 3 6 0 0 0 0 0 0 0 0 0]
            [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
            [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
            [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
            [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]]
        Output:
            [[6 6 3 6 6 6]
            [6 6 3 3 3 6]
            [6 3 6 6 6 6]
            [6 3 6 6 6 6]]
    
    Colour Encoding:
        Black = 0, Dark Blue = 1, Red = 2 , Green = 3 , Yellow = 4 , Grey = 5 , Pink = 6 , Orange = 7 , Sky Blue = 8 , Brown = 9
        
    Algorithm:
        Find the rectangle containing non-black(0) cells and then flip it along horizontal axis
        
    Results:
        All the 3 train test cases and 1 testing test cases passed
    """
    assert type(x) == np.ndarray
    coords = np.argwhere(x)                         # find coordinates for non-balck cells
    x_min, y_min = coords.min(axis=0)               # find coordinate of left-top cell
    x_max, y_max = coords.max(axis=0)               # find coordinate of right-bottom cell 
    cropped = x[x_min:x_max+1, y_min:y_max+1]       # get the croped cell, this will be rectangle with non-black cells
    return np.flip(cropped,1)                       # flip the rectangle along horizontal axis

################################################################################################################################

def solve_68b16354(x):
    """
    Task Description:
        Input:
            [[2 8 1 3 2 4 1]
            [4 4 1 1 4 3 4]
            [1 1 1 1 4 7 3]
            [1 1 2 3 8 1 3]
            [4 1 1 1 7 8 4]
            [3 2 8 4 1 8 4]
            [1 4 7 1 2 3 4]]
        Output:
            [[1 4 7 1 2 3 4]
            [3 2 8 4 1 8 4]
            [4 1 1 1 7 8 4]
            [1 1 2 3 8 1 3]
            [1 1 1 1 4 7 3]
            [4 4 1 1 4 3 4]
            [2 8 1 3 2 4 1]]
    
    Colour Encoding:
        Black = 0, Dark Blue = 1, Red = 2 , Green = 3 , Yellow = 4 , Grey = 5 , Pink = 6 , Orange = 7 , Sky Blue = 8 , Brown = 9
        
    Algorithm:
        The output matrix is mirror image of input matrix along vertical top axis
        
    Results:
        All the 3 train test cases and 1 testing test cases passed
    """
    assert type(x) == np.ndarray
    return np.flip(x,0) # flip the array vertically 

################################################################################################################################

def solve_c9f8e694(x):
    """
    Task Description:
        Input:
            [[0 0 0 0 0 0 0 0 0 0 0 0]
            [1 0 5 5 0 0 0 0 0 0 0 0]
            [2 0 5 5 0 0 0 0 0 0 0 0]
            [1 0 5 5 0 0 0 0 0 0 0 0]
            [1 0 5 5 0 0 0 0 0 0 0 0]
            [1 0 5 5 0 0 0 0 5 5 0 0]
            [2 0 5 5 0 0 0 0 5 5 0 0]
            [2 0 5 5 0 0 0 0 5 5 0 0]
            [1 0 0 0 0 0 0 0 5 5 0 0]
            [1 0 0 0 5 5 5 0 5 5 0 0]
            [1 0 0 0 5 5 5 0 5 5 0 0]
            [2 0 0 0 5 5 5 0 5 5 0 0]]
        Output:
            [[0 0 0 0 0 0 0 0 0 0 0 0]
            [1 0 1 1 0 0 0 0 0 0 0 0]
            [2 0 2 2 0 0 0 0 0 0 0 0]
            [1 0 1 1 0 0 0 0 0 0 0 0]
            [1 0 1 1 0 0 0 0 0 0 0 0]
            [1 0 1 1 0 0 0 0 1 1 0 0]
            [2 0 2 2 0 0 0 0 2 2 0 0]
            [2 0 2 2 0 0 0 0 2 2 0 0]
            [1 0 0 0 0 0 0 0 1 1 0 0]
            [1 0 0 0 1 1 1 0 1 1 0 0]
            [1 0 0 0 1 1 1 0 1 1 0 0]
            [2 0 0 0 2 2 2 0 2 2 0 0]]
    
    Colour Encoding:
        Black = 0, Dark Blue = 1, Red = 2 , Green = 3 , Yellow = 4 , Grey = 5 , Pink = 6 , Orange = 7 , Sky Blue = 8 , Brown = 9
        
    Algorithm:
        We need to fill all the grey cell with the first colour found in each row. This way we will generate a horizontal
        flowing matrix
        
    Results:
        All the 2 train test cases and 1 testing test cases passed
    """
    assert type(x) == np.ndarray
    height,width = x.shape                      # height and width of numpy 2D array
    x_answer = x.copy()                         # create a copy of original matrix
    for i in range(height):
        for j in range(1,width):                # exculding the first vertical column
            if x_answer[i][j]!=0:
                x_answer[i][j] = x_answer[i][0] # fill each non-black cell (here grey cell) with the first colour to its row found at (i,0)
    return x_answer

################################################################################################################################

def main():
    # Find all the functions defined in this file whose names are
    # like solve_abcd1234(), and run them.

    # regex to match solve_* functions and extract task IDs
    p = r"solve_([a-f0-9]{8})" 
    tasks_solvers = []
    # globals() gives a dict containing all global names (variables
    # and functions), as name: value pairs.
    for name in globals(): 
        m = re.match(p, name)
        if m:
            # if the name fits the pattern eg solve_abcd1234
            ID = m.group(1) # just the task ID
            solve_fn = globals()[name] # the fn itself
            tasks_solvers.append((ID, solve_fn))

    for ID, solve_fn in tasks_solvers:
        # for each task, read the data and call test()
        directory = os.path.join("..", "data", "training")
        json_filename = os.path.join(directory, ID + ".json")
        data = read_ARC_JSON(json_filename)
        test(ID, solve_fn, data)
    
def read_ARC_JSON(filepath):
    """Given a filepath, read in the ARC task data which is in JSON
    format. Extract the train/test input/output pairs of
    grids. Convert each grid to np.array and return train_input,
    train_output, test_input, test_output."""
    
    # Open the JSON file and load it 
    data = json.load(open(filepath))

    # Extract the train/test input/output grids. Each grid will be a
    # list of lists of ints. We convert to Numpy.
    train_input = [np.array(data['train'][i]['input']) for i in range(len(data['train']))]
    train_output = [np.array(data['train'][i]['output']) for i in range(len(data['train']))]
    test_input = [np.array(data['test'][i]['input']) for i in range(len(data['test']))]
    test_output = [np.array(data['test'][i]['output']) for i in range(len(data['test']))]

    return (train_input, train_output, test_input, test_output)


def test(taskID, solve, data):
    """Given a task ID, call the given solve() function on every
    example in the task data."""
    print(taskID)
    train_input, train_output, test_input, test_output = data
    print("Training grids")
    for x, y in zip(train_input, train_output):
        yhat = solve(x)
        show_result(x, y, yhat)
    print("Test grids")
    for x, y in zip(test_input, test_output):
        yhat = solve(x)
        show_result(x, y, yhat)

        
def show_result(x, y, yhat):
    print("Input")
    print(x)
    print("Correct output")
    print(y)
    print("Our output")
    print(yhat)
    print("Correct?")
    # if yhat has the right shape, then (y == yhat) is a bool array
    # and we test whether it is True everywhere. if yhat has the wrong
    # shape, then y == yhat is just a single bool.
    print(np.all(y == yhat))

if __name__ == "__main__": main()

