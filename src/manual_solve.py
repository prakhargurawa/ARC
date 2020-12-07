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
        All the 3 train test cases and 1 test cases passed
    """
    
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

