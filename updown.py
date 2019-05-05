"""
This module contains helper functions for analyzing the success of our model's predictions

"""

def up_down(stuff):
    binary = []
    for i in range(1, len(stuff)):
        if stuff[i] > stuff[i-1]:
            binary.append(1)
        elif stuff[i] < stuff[i-1]:
            binary.append(-1)
        else:
            binary.append(0)
    return binary

def compare_updown(first, second):
    correct = 0
    total = len(first)
    for i in range(len(first)):
        if first[i] == second[i]:
            correct += 1
    return correct, correct / total
