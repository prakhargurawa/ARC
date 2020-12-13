# The Abstraction and Reasoning Corpus (ARC)

### Assignment PTAI ###
Submitted By Prakhar Gurawa (20231064) and Sashwanth Sanjeev Kumar Sharma (20230331)
___

In the context of AI research, Legg and Hutter defined Intelligence as “Intelligence measures an agent’s ability to achieve goals in a wide range of environments.” One of the biggest concerns in AI currently is the field on General Intelligence, where a single program is able to perform or tackle most of the tasks without any external human interference.

For this assignment we have restricted our approach to Task specific, where we are relying on hand-coded solution for each specific task. But the solution can be extended to broad generalization and extreme generalization as future work. 

We have solved about 14 tasks but will be describing the below tasks for major consideration for grading:

#### 1. Task 846bdb03 : 
This is like a puzzle game where we need to connect same colour blocks to same ends. If the two different colour blocks are aligned with the colour of ends, they will exactly be copied to attached to ends otherwise the blocks need to be flipped and attached to the ends. The sizes of puzzle blocks are not fixed so our output matrix width will depend on them and height on the two handles present in diagram. 

![alt text](https://github.com/prakhargurawa/ARC/blob/feature_arc/images/846bdb03.png?raw=true)

#### 2. Task 007bbfb7
The output matrix's dimension is square of input matrix's dimension. The output matrix gets a copy of full input matrix if the cell in input matrix in non-black (0). For efficient copy we have used numpy slicing technique.

![alt text](https://github.com/prakhargurawa/ARC/blob/feature_arc/images/007bbfb7.png?raw=true)

#### 3. Task d07ae81c
We are provided with a 2D numpy matrix with has 2 backgrouds and some outliers, the output matrix we need to change colour of all cells which are in diagonal of those outliers. We first need to find outliers (used isOutlier function for this) and save the colour mapping which will be used to change the colour of cell later. The isOutlier function returns the colour which will be changed to colour of outlier later.

![alt text](https://github.com/prakhargurawa/ARC/blob/feature_arc/images/d07ae81c.png?raw=true)

#### 4. Task f35d900a
The description of the task is, given 2 pairs of any colours (only 2 unique colours at a time) at any position but each cell would be in the same row and column with the other 2 cells and almost diagonal to 1 cell, so there would be total 4 cells with 2 unique colours is the shape of square or rectangle, so let’s call these cells as centre cells. The goal of the task is 1. to colour the neighbouring cells of each centre cell with given opposite colour, and 2. for given 4 centre cells, fill the row and column alternate cells with grey (5) for each centre cell only to half distance to the other neighbouring centre cells with a condition of coloured cells should not be next to each other in certain cases and can exists next to each other in certain condition. 

![alt text](https://github.com/prakhargurawa/ARC/blob/feature_arc/images/f35d900a.png?raw=true)

#### 5. Task ae3edfdc
The description of the task is, there would be any different colours (0-9) in the given matrix, and some colour would be centre colour (or one-time occurrence in the matrix). For the centre colour, there may be another colour in the same row and/or column (its occurrence would be more than one), so let’s call that colour as pair colour. For every centre colour, the respective pair colours should move towards the centre and should be placed to neighbouring indices of its centre colour, but the direction of the pair colour should be maintained, if the pair colour was on right side of the centre colour then the pair colour should be placed right side neighbouring indices of the centre colour and respectively other positions.

![alt text](https://github.com/prakhargurawa/ARC/blob/feature_arc/images/ae3edfdc.png?raw=true)

#### 6. Task f2829549
The problem asks us to find non-black cells on the left and right side of pillar formed using dark blue colour. We will first divide the array into two parts to find left and right matrix based on width of original matrix.


![alt text](https://github.com/prakhargurawa/ARC/blob/feature_arc/images/f2829549.png?raw=true)

## Summary on python features and libraries used:
Almost every task is solved using python numpy library since it involves several mathematical calculations, and excellent features of numpy is its performance, size and functionality, while performing some mathematical calculations, it is not necessary to loop through rows and columns (because it is vectorised), instead use the numpy built-in functions which facilitates the need with faster approach, some of the functions which are more commonly used in the tasks are (given below are basic syntax and can be extended further based on the requirement)

    I. np.where ( x == value), where x is the 2D numpy array and y is the value of the cell.This function will return all the indices of the matrix x for the given value
    
    II. np.argwhere(x) or np.argwhere(<x - condition>), where x is the 2D numpy array.This function will return all the indices whose value is non-zero and takes conditional input also.
    
    III. np.unique(x), where x is the 2D numpy array.This function will return all the unique values available in the given matrix x.
    
    IV. np.multiply(x1,x2), where x1 and x2 are 2D numpy array with same dimensions.This function will multiply the values of similar indices of both the matrix and return a matrix with multiplied values at respective position/cells.
    
    V. np.flip(x, axis=0 or 1), where x is the 2D numpy array.This function flips the array either vertically or horizontally based on the axis value. 
    
    VI. np. diag_indices(n), where n is the dimension of n x n matrix.This function will return all the diagonal index for the matrix with size n x n
    
    
## Similarities:

1.	In ARC, all the tasks are related to matrix manipulation and understanding the pattern or the relationship between the input and output of the task is the most important factor.

2.	There are group of tasks which has similar pattern, but the similarity would be only for a subset or part of the pattern, for instance simple task pattern exists in medium and difficult task also but as a subset pattern.

3.	For these kinds of tasks same approach can be used to solve the pattern, i.e., these subsets of problems can be divided into smaller or simpler task and device an approach of developing micro functions that could solve similar smaller problems and help us to reduce the redundancy of the code and lead us to the path of broad generalization.

4.	These micro functions can be used as input to the actual high-level task and solve the general patterns/problems.

## Differences:

1.	There are tasks where new elements are not added to input matrix, for example: task (846bdb03), whereas there are tasks where we need some kind of manipulation and add new components to input matrix and perform conversion to output matrix, for example (d07ae81c).

2.	There are tasks where the input and output have different dimensions, where the input of the matrix is not modified but taken as reference and also the tasks with same dimension where modification happens on the same matrix itself.


3.	There are different categories of task which has completely different approach of solving the pattern

    * Some of them are flipping the matrix or having half mirror image of the other side or foldable pattern. 
    * Tasks like recognising a small pattern and develop a big picture of it into multiple small patterns with some rules and similarly to recognise a small pattern from a big         picture.
    * Tasks like puzzles, like placing a subset of pattern inside another pattern with some rules and placing all the patterns one above the other and deriving a final pattern.

These kind of different problems or patterns is a great approach to train a model and test its general intelligence of solving the pattern
___

This repository contains the ARC task data, as well as a browser-based interface for humans to try their hand at solving the tasks manually.

*"ARC can be seen as a general artificial intelligence benchmark, as a program synthesis benchmark, or as a psychometric intelligence test. It is targeted at both humans and artificially intelligent systems that aim at emulating a human-like form of general fluid intelligence."*

A complete description of the dataset, its goals, and its underlying logic, can be found in: [The Measure of Intelligence](https://arxiv.org/abs/1911.01547).

As a reminder, a test-taker is said to solve a task when, upon seeing the task for the first time, they are able to produce the correct output grid for *all* test inputs in the task (this includes picking the dimensions of the output grid). For each test input, the test-taker is allowed 3 trials (this holds for all test-takers, either humans or AI).


## Task file format

The `data` directory contains two subdirectories:

- `data/training`: contains the task files for training (400 tasks). Use these to prototype your algorithm or to train your algorithm to acquire ARC-relevant cognitive priors.
- `data/evaluation`: contains the task files for evaluation (400 tasks). Use these to evaluate your final algorithm. To ensure fair evaluation results, do not leak information from the evaluation set into your algorithm (e.g. by looking at the evaluation tasks yourself during development, or by repeatedly modifying an algorithm while using its evaluation score as feedback).

The tasks are stored in JSON format. Each task JSON file contains a dictionary with two fields:

- `"train"`: demonstration input/output pairs. It is a list of "pairs" (typically 3 pairs).
- `"test"`: test input/output pairs. It is a list of "pairs" (typically 1 pair).

A "pair" is a dictionary with two fields:

- `"input"`: the input "grid" for the pair.
- `"output"`: the output "grid" for the pair.

A "grid" is a rectangular matrix (list of lists) of integers between 0 and 9 (inclusive). The smallest possible grid size is 1x1 and the largest is 30x30.

When looking at a task, a test-taker has access to inputs & outputs of the demonstration pairs, plus the input(s) of the test pair(s). The goal is to construct the output grid(s) corresponding to the test input grid(s), using 3 trials for each test input. "Constructing the output grid" involves picking the height and width of the output grid, then filling each cell in the grid with a symbol (integer between 0 and 9, which are visualized as colors). Only *exact* solutions (all cells match the expected answer) can be said to be correct.


## Usage of the testing interface

The testing interface is located at `apps/testing_interface.html`. Open it in a web browser (Chrome recommended). It will prompt you to select a task JSON file.

After loading a task, you will enter the test space, which looks like this:

![test space](https://arc-benchmark.s3.amazonaws.com/figs/arc_test_space.png)

On the left, you will see the input/output pairs demonstrating the nature of the task. In the middle, you will see the current test input grid. On the right, you will see the controls you can use to construct the corresponding output grid.

You have access to the following tools:

### Grid controls

- Resize: input a grid size (e.g. "10x20" or "4x4") and click "Resize". This preserves existing grid content (in the top left corner).
- Copy from input: copy the input grid to the output grid. This is useful for tasks where the output consists of some modification of the input.
- Reset grid: fill the grid with 0s.

### Symbol controls

- Edit: select a color (symbol) from the color picking bar, then click on a cell to set its color.
- Select: click and drag on either the output grid or the input grid to select cells.
    - After selecting cells on the output grid, you can select a color from the color picking to set the color of the selected cells. This is useful to draw solid rectangles or lines.
    - After selecting cells on either the input grid or the output grid, you can press C to copy their content. After copying, you can select a cell on the output grid and press "V" to paste the copied content. You should select the cell in the top left corner of the zone you want to paste into.
- Floodfill: click on a cell from the output grid to color all connected cells to the selected color. "Connected cells" are contiguous cells with the same color.

### Answer validation

When your output grid is ready, click the green "Submit!" button to check your answer. We do not enforce the 3-trials rule.

After you've obtained the correct answer for the current test input grid, you can switch to the next test input grid for the task using the "Next test input" button (if there is any available; most tasks only have one test input).

When you're done with a task, use the "load task" button to open a new task.
