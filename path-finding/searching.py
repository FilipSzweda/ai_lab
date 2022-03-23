import sys

from maze import Maze, path_from

def getCost(curr):
    return curr.cost

def getPriority(curr):
    return curr.priority

def estDist(curr):
    end_node = maze.find_node('E')
    return abs(curr.x-end_node.x)+abs(curr.y-end_node.y)


def bfs(maze):
    start_node = maze.find_node('S')
    q = [start_node]
    while len(q) > 0:
        node = q.pop(0)  # FIFO
        node.visited = True
        if node.type == 'E':
            return path_from(node)

        children = maze.get_possible_movements(node)
        for child in children:
            if not child.visited:
                child.parent = node
                q.append(child)

    return None


def dijikstra(maze):
    start_node = maze.find_node('S')
    start_node.cost = 0
    q = [start_node]
    while len(q) > 0:
        q.sort(reverse=False, key=getCost)
        node = q.pop(0)  # FIFO
        node.visited = True
        if node.type == 'E':
            return path_from(node)
        children = maze.get_possible_movements(node)
        for child in children:
            new_cost = node.cost+maze.move_cost(node, child)
            if new_cost < child.cost:
                child.cost = new_cost
                child.parent = node
                q.append(child)

    return None

def a_star(maze):
    start_node = maze.find_node('S')
    start_node.cost = 0
    start_node.priority = 0
    q = [start_node]
    while len(q) > 0:
        q.sort(reverse=False, key=getPriority)
        node = q.pop(0)  # FIFO
        node.visited = True
        if node.type == 'E':
            return path_from(node)
        children = maze.get_possible_movements(node)
        for child in children:
            new_cost = node.cost+maze.move_cost(node, child)
            if new_cost < child.cost:
                child.cost = new_cost
                child.priority = child.cost + estDist(child)
                child.parent = node
                q.append(child)

    return None

def gbfs(maze):
    start_node = maze.find_node('S')
    q = [start_node]
    while len(q) > 0:
        q.sort(reverse=False, key=estDist)
        node = q.pop(0)  # FIFO
        node.visited = True
        if node.type == 'E':
            return path_from(node)

        children = maze.get_possible_movements(node)
        for child in children:
            if not child.visited:
                child.parent = node
                q.append(child)

    return None


algorithm = sys.argv[1]
maze = Maze.from_file(sys.argv[2])
maze.draw()
if algorithm == "bfs":
    maze.path = bfs(maze)
elif algorithm == "dijikstra":
    maze.path = dijikstra(maze)
elif algorithm == "gbfs":
    maze.path = gbfs(maze)
elif algorithm == "a_star":
    maze.path = a_star(maze)
print()
maze.draw()
print('path length: ', len(maze.path))
for node in maze.path:
    print(f'({node.x}, {node.y})', end=' ')
print()