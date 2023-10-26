def find_minimum_k(N, trees, M, sprinklers):
    gaps = []
    
    # Calculate the gaps between adjacent trees
    for i in range(1, N):
        gaps.append(abs(trees[i] - trees[i - 1]) - 1)

    # Sort the sprinklers and consider the gaps between trees and the sprinklers
    sprinklers.sort()

    k_values = []
    for gap in gaps:
        k_values.append((gap + 1) // 2)

    # Consider the distances from the first and last trees to the nearest sprinklers
    k_values.append(sprinklers[0] - trees[0])
    k_values.append(trees[-1] - sprinklers[-1])

    return max(k_values)

# Read input
N = int(input())
trees = list(map(int, input().split()))
M = int(input())
sprinklers = list(map(int, input().split()))

# Find and print the minimum value of k
print(find_minimum_k(N, trees, M, sprinklers))
