from math import sqrt
from typing import List, Tuple
import numpy as np



def read_data(file):
    """
    Read a text file of 2D points and returns a list like [(x1, y1), (x2, y2), ..., (xN, yN)]
    
    :param file: File in the proper format giving all the coordinates of the cities 
    """
    stream = open(file)
    lines = stream.readlines()
    coordinates = []
    for line in lines:
        parts = line.split(" ")
        coordinates.append(((float(parts[0]), float(parts[1]))))
    return coordinates


def distance(p1, p2):
    """
    Standard Euclidian distance between 2 points p1 and p2, will later become the edge cost between 2 cities. 
    
    :param p1: 2D point 1.
    :param p2: 2D point 2.
    """
    return sqrt(pow(p1[0] - p2[0], 2) + pow(p1[1] - p2[1], 2))


def distance_matrix(coordinates: List[Tuple]):
    """
    Creates the distance matrix between all the cities (N x N) matrix containing the N cities. 
    
    :param coordinates: List of all the coordinates of all the cities 
    """
    N = len(coordinates)
    matrix = np.zeros((N, N))
    for i in range(0, N):
        cord1 = coordinates[i] 
        for j in range(0, N):
            cord2 = coordinates[j]
            matrix[i][j] = distance(cord1, cord2)
    return matrix


def get_largest(matrix):
    """
    Returns the maximum distance in the distance matrix 
    
    :param matrix: Description
    """
    largest = 0.0
    N = len(matrix)
    for i in range(0, N):
        for j in range(0, N):
            largest = largest if largest > matrix[i][j] else matrix[i][j]
    return largest


def normalize(matrix):
    largest = get_largest(matrix)
    N = len(matrix)
    for x in range(0, N):
        for y in range(0,N):
            matrix[x][y] /= largest
    return matrix


def normalize_cords(coordinates):
    N = len(coordinates)
    xs = np.zeros(N)
    ys = np.zeros(N)
    for i in range(0, N):
        xs[i] = coordinates[i][0]
        ys[i] = coordinates[i][1]
    largest_x = max(xs)
    largest_y = max(ys)

    for pos in range(0, N):
        xs[pos] /= largest_x
        ys[pos] /= largest_y

    return list(zip(xs, ys))



