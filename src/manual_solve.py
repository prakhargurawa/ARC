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

def solve_846bdb03(x):
    
    """
    Task Description:
        Input:
        [[0 0 0 0 0 0 0 0 0 0 0 0 0]
        [0 0 0 0 0 0 0 0 0 0 0 0 0]
        [0 0 0 0 0 0 0 0 0 0 0 0 0]
        [0 0 2 2 0 1 0 0 0 0 0 0 0]
        [0 0 0 2 0 1 1 1 0 0 0 0 0]
        [0 0 0 2 2 1 0 0 0 0 0 0 0]
        [0 0 0 0 2 0 0 0 0 0 0 0 0]
        [0 0 0 0 0 4 0 0 0 0 0 0 4]
        [0 0 0 0 0 2 0 0 0 0 0 0 1]
        [0 0 0 0 0 2 0 0 0 0 0 0 1]
        [0 0 0 0 0 2 0 0 0 0 0 0 1]
        [0 0 0 0 0 2 0 0 0 0 0 0 1]
        [0 0 0 0 0 4 0 0 0 0 0 0 4]]
        
        Desired Output:
        [[4 0 0 0 0 0 0 4]
        [2 2 2 0 1 0 0 1]
        [2 0 2 0 1 1 1 1]
        [2 0 2 2 1 0 0 1]
        [2 0 0 2 0 0 0 1]
        [4 0 0 0 0 0 0 4]]
    
    Colour encodings:
        Black = 0, Dark Blue = 1, Red = 2 , Green = 3 , Yellow = 4 , Grey = 5 , Pink = 6 , Orange = 7 , Sky Blue = 8 , Brown = 9
    
    Algorithm:
        This is like a puzzle game where we need to connect same color blocks to same ends. If the two different color blocks are alligned 
        with the colour of ends, they will exactly copied to attached to ends otherwise the blocks need to be flipped and attached
        to the ends. The sizes of puzzle blocks are not fixed so our output matrix width will depend on them and height on the two 
        handles present in diagram.
        
    Implementation:
        Find first the yellow cells and the the two colors on pillar .Also remember that which colour is at left and which at right.
        Remove the two poles from input matrix and find the two puzzle pieces by cropping the segment after finding x_min,y_min,x_max,y_max
        of each colour. Check whether the two colour puzzle are alligned with colour of poles and copy exact puzzle or flipped image of
        puzzle accordingly.
        
    Results:
        All the 4 train test cases and 1 testing test cases passed
    """
    assert type(x) == np.ndarray    
    x_copy = x.copy()                               # keep a copy of original matrix
    Z = np.argwhere(x==4)                           # find the position of yellow (4) cells , returns a list of tuple of positions of all yellow cells (which are always 4) 
    height = Z[2][0] - Z[0][0] +1                   # the height of output matrix can be calculated by positions of yellow cells

    color_left = x[Z[0][0]+1][Z[0][1]]              # colour on left pole
    color_right = x[Z[1][0]+1][Z[1][1]]             # colour on right pole

    x_copy[Z[0][0]:Z[3][0]+1,Z[0][1]] = 0           # mark left pole as black (0) ,useful for future calculations
    x_copy[Z[0][0]:Z[3][0]+1,Z[1][1]] = 0           # mark right pole as black(0) ,useful for future calculations
    
    x_color_left = np.where(x_copy==color_left,1,0) # craete a matrix where left color is 1 rest 0
    coords = np.argwhere(x_color_left)              # find all coordinates of 1 (here left color)
    x_min_left, y_min_left = coords.min(axis=0)     # min x,min y of 1
    x_max_left, y_max_left = coords.max(axis=0)     # max x,max y of 1  
    x_color_left_part = x_copy[x_min_left:x_max_left+1, y_min_left:y_max_left+1] # crop the puzzle part with left colour

    x_color_right = np.where(x_copy==color_right,1,0)   # craete a matrix where right color is 1 rest 0
    coords = np.argwhere(x_color_right)                 # find all coordinates of 1 (here right color)
    x_min_right, y_min_right = coords.min(axis=0)       # min x,min y of 1
    x_max_right, y_max_right = coords.max(axis=0)       # max x,max y of 1 
    x_color_right_part = x_copy[x_min_right:x_max_right+1, y_min_right:y_max_right+1] # crop the puzzle part with right colour
    
    width = x_color_left_part.shape[1] + x_color_right_part.shape[1] +2 # the width of output matrix depends on the two puzzle blocks +2 (for the two yellow cells)
    
    x_answer = np.zeros((height,width))                 # create a output matrix 
    x_answer[0][0] = x_answer[height-1][0] = x_answer[0][width-1] = x_answer[height-1][width-1] = 4 # mark all corners as yellow (4) cell
    x_answer[1:height-1,0] = color_left                 # mark left border with colour left
    x_answer[1:height-1,width-1] = color_right          # mark right border with colout right
    
    if y_max_right > y_max_left: # Alligned case
        # If case is alligned we need to as it copy the two puzzle blocks between poles
        x_answer[1:1+x_color_left_part.shape[0],1:1+x_color_left_part.shape[1]] = x_color_left_part
        base = x_color_left_part.shape[1] +1
        x_answer[1:1+x_color_right_part.shape[0],base:base+x_color_right_part.shape[1]] = x_color_right_part
    else: # Non- alligned case
        # If case is non-alligned we need to as it copy the two puzzle blocks between poles after flipping the puzzle blocks
        x_answer[1:1+x_color_left_part.shape[0],1:1+x_color_left_part.shape[1]] = np.flip(x_color_left_part,1) # flip left puzzle block
        base = x_color_left_part.shape[1] +1
        x_answer[1:1+x_color_right_part.shape[0],base:base+x_color_right_part.shape[1]] = np.flip(x_color_right_part,1) # flip right puzzle block
    
    return x_answer.astype(int)

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
        
    Implementation:
        Create output matrix with dimension squared of dimension of input matrix. Find indexes of non zero cells using np.nonzero
        For each non zero cell index copy the exact original cell to putput cell considering the indexes properly.
        
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

def isSafe(x,y,shape):
    height,width = shape
    if x>=0 and x<height and y>=0 and y<width:
        return True
    return False

def isOutlier(A,i,j):
    # This function finds number of neighbour of a cell and counts of unique colour of neighbour which are not as same colour as the cell
    D = defaultdict(int)
    num_neighbour = 0
    # Representation of 8 possible neighbour of a cell
    X = [-1, 0, 1, 1, 1, 0,-1,-1]       # possible value of increment in x coordinate to search for neighbours
    Y = [-1,-1,-1, 0, 1, 1, 1, 0]       # possible value of increment in y coordinate to search for neighbours
    
    for x,y in zip(X,Y):
        if isSafe(i+x,j+y,A.shape):
            num_neighbour +=1
            if A[i+x][j+y]!=A[i][j]:
                D[A[i+x][j+y]] += 1
      
    if sum(D.values()) == num_neighbour:# if all the neighbours have different colour that the cell then its a outlier
        inverse = [(value, key) for key, value in D.items()]
        return max(inverse)[1]          # we will return the mapping colour as the colour which occured most time, solved on paper this works well
    return -1                           # return -1 if its not an outlier
 
        
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
        
    Implementation:
        Find all outliers using isOutlier (The function find number of neighour of input cell and their colour and using that info finds
        whether input cell is outlier or not).Later for each outler change colour of all cells diagonal to that outlier cell.
        
    Results:
        All the 3 train test cases and 1 testing test cases passed
    """
    assert type(x) == np.ndarray    
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

def solve_ae3edfdc(x):
    
    """ 
    Task Description:
        Input:
            [[0 0 0 7 0 0 0 0 0 0 0 0 0 0 0]
             [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
             [7 0 0 1 0 0 0 0 0 7 0 0 0 0 0]
             [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
             [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
             [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
             [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
             [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
             [0 0 0 7 0 0 0 0 0 0 0 0 0 0 0]
             [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
             [0 0 0 0 0 3 0 0 0 0 0 2 0 0 0]
             [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
             [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
             [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
             [0 0 0 0 0 0 0 0 0 0 0 3 0 0 0]]
        Output:
            [[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
             [0 0 0 7 0 0 0 0 0 0 0 0 0 0 0]
             [0 0 7 1 7 0 0 0 0 0 0 0 0 0 0]
             [0 0 0 7 0 0 0 0 0 0 0 0 0 0 0]
             [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
             [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
             [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
             [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
             [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
             [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
             [0 0 0 0 0 0 0 0 0 0 3 2 0 0 0]
             [0 0 0 0 0 0 0 0 0 0 0 3 0 0 0]
             [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
             [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
             [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]]
    
    Colour Encoding:
        Black = 0, Dark Blue = 1, Red =2 , Green = 3 , Yellow = 4 , Grey = 5 , Pink = 6 , Orange = 7 , Sky Blue = 8 , Brown = 9
        
    Algorithm:
        The description of the task is, there would be any different colours(0-9) in the given matrix, and some colour would be center colour (or one time occurence in the matrix). \
        For the center color, there may be another colour in the same row and/or column (its occurence would be more than one), so lets call that colour as pair colour. \
        For every center colour, the respective pair colours should move towards the centre and should be placed to neighbouring indices of the its center colour, \
        but the direction of the pair colour should be maintained, if the pair color was on right side of the center colour then the pair colour should be placed right side neighbouring indices of the center colour and respectively other positions.
    
    Implementation:        
        First all the unique colours in the matrix are fetched except black(0), iterating through each colour and find their indices if the length of the indices is equal to one then it is a center colour, if the lenghth of the indices is more than one then it is pair color. \
        Filtering through center colours and fetch the whole colours again and filter all the pair colours (if length of the indices is more than one) for comparasion. \
        Each center colour is compared with pair colours and finding whether the pair colour exists on the same row or same column, if the pair colour exists in the same row or same column, flip the pair colour to black and then check which side of the center color it exists using source matrix(x) . \
            
        If the pair colour row index is more than the center colour row index, then the position is bottom.
        If the pair colour row index is less than the center colour row index, then the position is top.
        If the pair colour column index is more than the center colour column index, then the position is right.
        If the pair colour column index is less than the center colour column index, then the position is left.
        
        After finding the position, from the center colour either column indices or row indices are adjusted to +1 or -1 based on the above if conditions and flipped it the respective pair colour.
              
    Results:
        All the 3 train test cases and 1 testing test cases passed
    """ 
    assert type(x) == np.ndarray 
    x_copy = x.copy()
    unique = np.unique(x)[1:]                                                  # all the unique colours are fetched from the matrix
    for center in unique:                                                      # iterating through each colour 
        row, column = np.where(x == center)                                    # fetch the indices of the colour
        # focusing on the center point
        if len(row) == 1 and len(column) == 1:                                 # if the length of the indices is equal to one, then it is a center colour
            for pair in unique:                                                # parsing the colours again to compare and find its respective pair colour
                p_row, p_column = np.where(x == pair)                          # fetch the indices of the color
                # Focusing on the pair points
                if center != pair and len(p_row) != 1 and len(p_column) != 1:  # filter out if the center colour and pair colour are same, then filter through the colours whose length of indcies are more than one. ie., pair colour
                    for p_r, p_c in zip(p_row,p_column):                       # since it is pair colour, above np.where would return more than one row and column indcies, i.e., parse through each pair colour.
                        if p_r == row:                                         # check whether the pair colour is in same row for the respective center colour and then check for which position or side 
                            x_copy[p_r][p_c] = 0                               # flip the actual cell to black
                            if p_c < column:                                   # check if pair colour column index is less than center colour column index
                                x_copy[p_r][column-1] = pair                   # if yes, then the position/side of the pair colour is left so subtracting center colour column index with -1 and flip the colour of the cell to respective iteration of the colour(pair)
                            elif p_c > column:                                 # extra check inorder to aviod unnecesary matrix alteration, check whether the pair colour column index is more than the center colour column index
                                x_copy[p_r][column+1] = pair                   # if yes, then the position/side of the pair colour is right so adding center colour column index with +1 and flip the colour of the cell to respective iteration of the colour(pair)
                        if p_c == column:                                      # check if the pair colour is in the same column
                            x_copy[p_r][p_c] = 0                               # flip the actual cell to black
                            if p_r < row:                                      # check if pair colour row index is less than center colour row index
                                x_copy[row[0]-1][p_c] = pair                   # if yes, then the position/side of the pair colour is top so subtracting center colour row index with -1 and flip the colour of the cell to respective iteration of the colour(pair)
                            elif p_r > row:                                    # extra check inorder to aviod unnecesary matrix alteration, check whether the pair colour row index is more than the center colour row index
                                x_copy[row[0]+1][p_c] = pair                   # if yes, then the position/side of the pair colour is bottom so adding center colour row index with +1 and flip the colour of the cell to respective iteration of the colour(pair)
    return x_copy


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
    
    Implementation:
        Use argwhere to find indexes of red cell, this will give us idea of width of red cell pole. Later use the width to fill
        Green and blue poles by increasing and decreasing width sizes respectively. Start thiese operation index just above and below
        of index of red cell.
        
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
        
    Implemenation:
        If all the cell colur in first row are same than this will be a horizontal flowing matrix otherwise vertical. If it is 
        horizontal flowing matrix traverse the first column and store all unique colour and store in list data structure. Finally
        return as numpy array.
        
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
        The problem ask us to find non-black cells on the left and right side of pillar formed using dark blue colour. We will
        first divide the array into two parts to find left and right matrix based on width of original matrix.
        
    Implementation:
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
        The problem asks us to find most unique 3x3 matrix cell ignoring the colour scheme. Created an utility function find_different
        to find the most unique cell ifnoring colour scheme.
        
    Impelmentation:
        We will first split the given 2D numpy in 3x3 matrix and will find the matrix which is different from all other matrix.
        The split will be vertical if heigh is more than width else split will be horizontal. Vertical split is done using vsplit
        and horizontal split as hsplit.
        
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
    
    Implementation:
        Use argswhere to find coordinate of non-black cell and fetch x_min,y_min,x_max,y_max and use there coordinates to crop
        the non-black block.
        
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
        
    Implementation:
        For each cell in first column get the colour and fill same colur to all non-black cell of its row.
        
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

def solve_3bd67248(x):
    
    """ 
    Task Description:
        Input:
            [[8 0 0 0 0 0 2]
             [8 0 0 0 0 2 0]
             [8 0 0 0 2 0 0]
             [8 0 0 2 0 0 0]
             [8 0 2 0 0 0 0]
             [8 2 0 0 0 0 0]
             [8 4 4 4 4 4 4]]
        Output:
            [[8 0 0 0 0 0 2]
             [8 0 0 0 0 2 0]
             [8 0 0 0 2 0 0]
             [8 0 0 2 0 0 0]
             [8 0 2 0 0 0 0]
             [8 2 0 0 0 0 0]
             [8 4 4 4 4 4 4]]
    
    Colour Encoding:
        Black = 0, Dark Blue = 1, Red =2 , Green = 3 , Yellow = 4 , Grey = 5 , Pink = 6 , Orange = 7 , Sky Blue = 8 , Brown = 9
        
    Algorithm:
        The description of the task is, given a matrix with left most column filled completely with any colour (0-9), \
        the goal is to fill the last row of the matrix with yellow colour (4), except the first cell in the row and \
        fill the diagonal cells of the matrix from left bottom corner to top right corner with red colour (2) except the bottom left corner cell.
    
    Implementation:
        First the left most column colour is fetched for reference, then fill the last row of the matrix with yellow (4) \ 
        and to get the indices of the diagonal cells of the matrix, since the diagonal needed is from bottom left corner to top right corner, \
        the matrix is flipped upside down and diagonal cell indices (top left corner to bottom right corner) are fetched and filled with red (2) and inverted back to original position, \
        now the left bottom most corner cell is filled with the reference colour which is fetched initially.
              
    Results:
        All the 3 train test cases and 1 testing test cases passed
    """

    assert type(x) == np.ndarray 
    rows, columns = x.shape                             # fetching number of rows and columns in the given matrix for matrix manipulation
    left_col_color = x[-1:,0:1].copy()                  # fetching the left most column of the matrix colour for reference
    x[-1:,:] = 4                                        # filling the last row of the matrix with yellow(4)
    di = np.diag_indices(rows)                          # fetching the diagonal cell indices 
    x_flip = np.flipud(x)                               # flipping the matrix 
    x_flip[di] = 2                                      # filling the digonal cells with red(2) top left corner to bottom right corner
    x = np.flipud(x_flip)                               # on flipping back the array the top left corner would become bottom left corner, similarly botton right corner to top right corner
    x[-1:,0:1] = left_col_color[0][0]                   # filling the left most bottom cell with reference colour which was changed during diagonal colouring

    return x

################################################################################################################################

def solve_c1d99e64(x):
    
    """
    Task Description:
        Input:
            [[8 8 8 8 0 8 8 8 8 8 0 0 8 8]
             [0 8 0 0 0 0 8 8 8 8 0 8 8 8]
             [8 8 0 8 0 8 8 8 8 8 0 0 8 8]
             [8 0 8 8 0 8 8 0 0 8 0 8 8 0]
             [8 8 8 8 0 8 8 0 0 0 0 8 8 8]
             [8 8 8 0 0 8 8 0 8 0 0 8 8 8]
             [8 0 8 8 0 8 8 8 8 8 0 0 0 8]
             [8 8 0 0 0 8 0 0 8 8 0 0 8 8]
             [8 0 0 8 0 8 8 8 0 8 0 8 8 8]
             [8 8 0 8 0 8 8 8 8 8 0 0 8 0]
             [0 8 0 8 0 0 0 0 0 0 0 8 0 8]
             [8 8 8 8 0 8 8 8 8 8 0 0 8 0]]
        Output:
            [[8 8 8 8 2 8 8 8 8 8 2 0 8 8]
             [0 8 0 0 2 0 8 8 8 8 2 8 8 8]
             [8 8 0 8 2 8 8 8 8 8 2 0 8 8]
             [8 0 8 8 2 8 8 0 0 8 2 8 8 0]
             [8 8 8 8 2 8 8 0 0 0 2 8 8 8]
             [8 8 8 0 2 8 8 0 8 0 2 8 8 8]
             [8 0 8 8 2 8 8 8 8 8 2 0 0 8]
             [8 8 0 0 2 8 0 0 8 8 2 0 8 8]
             [8 0 0 8 2 8 8 8 0 8 2 8 8 8]
             [8 8 0 8 2 8 8 8 8 8 2 0 8 0]
             [0 8 0 8 2 0 0 0 0 0 2 8 0 8]
             [8 8 8 8 2 8 8 8 8 8 2 0 8 0]]
    
    Colour Encoding:
        Black = 0, Dark Blue = 1, Red =2 , Green = 3 , Yellow = 4 , Grey = 5 , Pink = 6 , Orange = 7 , Sky Blue = 8 , Brown = 9
        
    Algorithm:
        The description of the task is simple, the rows with all the cells black(0) and/or the columns with all the cells black(0) should be filled with colour red(2)
    
    Implementation:
        First fetch all the row indicies ('rows_black') which has all the cells in them as black(0) and similarly for columns('columns_black'), \
        Iterate through 'columns_black' to fill all the cells with red(2) and likewise iterate through 'columns_black' to fill all the cells with red(2)
              
    Results:
        All the 3 train test cases and 1 testing test cases passed
    """ 
    
    assert type(x) == np.ndarray 
    x_copy = x.copy()
    rows_black = np.where(~x_copy.any(axis=1))[0]              # fetch all the row indices which has all the cells with black
    columns_black = np.where(~x_copy.any(axis=0))[0]           # fetch all the column indicies which has all the cells with black
    for row in rows_black:                                     # iterate through row indices and fill the colour red(2) for rows
        x_copy[row,:] = 2
    for col in columns_black:                                  # iterate through column indices and fill the colour red(2) for columns
        x_copy[:,col] = 2    
    return x_copy
    
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

