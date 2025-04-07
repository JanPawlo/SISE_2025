import board as brd
import time



def bfs(board, order, max_iter=10000000):
    
    visited = set()
    queue = [(board.deepcopy(), [], 0)]
    i = 0
    highest_depth = 0
    start = time.time()

    while queue and i < max_iter:
        i += 1
        current_board, path, depth = queue.pop(0)
        state = tuple(tuple(i) for i in current_board._board)

        if state in visited:
            continue
        
        if depth > highest_depth:
            highest_depth = depth

        visited.add(state)

        if current_board._board == current_board._target_board:
            return path, len(visited), i, highest_depth, round((time.time() - start)*1000, 3)

        for move in current_board.find_possible_moves(order):
            new_board = current_board.deepcopy()
            new_board.make_move(move)
            queue.append((new_board, path + [move], depth+1))

       

    return -1, len(visited), i, highest_depth, round((time.time() - start)*1000, 3)


def dfs(board, order, max_iter=10000000, max_depth=20):
    visited = set()
    stack = [(board.deepcopy(), [], 0)]  # Store (board, path)
    i = 0
    order = order
    highest_depth = 0
    start = time.time()

    while stack and i < max_iter:
        
        current_board, path, depth = stack.pop()
        
        
        if depth > max_depth:
            continue
        
        state = tuple(tuple(i) for i in current_board._board)

        if state not in visited:
            visited.add(state)
            
        if depth > highest_depth:
            highest_depth = depth

        if current_board._board == current_board._target_board:
            return path, len(visited), i, highest_depth, round((time.time() - start)*1000, 3)

        for move in reversed(current_board.find_possible_moves(order)):
            new_board = current_board.deepcopy()
            new_board.make_move(move)
            stack.append((new_board, path + [move], depth+1))  # Add move to path

        i += 1


    return -1, len(visited), i, highest_depth, round((time.time() - start)*1000, 3)

def a_star(board, heuristic):
    
    openList = [(board.deepcopy(), [], 0)] # to be visited
    visited = set() # Already visited
    
    highest_depth = 0
    start = time.time()
    i = 0
    
    if heuristic == 'manh':
        heuristic = lambda x : x.manhattan_heuristic()
    elif heuristic == 'hamm':
        heuristic = lambda x : x.hamming_heuristic()

            
        
        
    while openList:
        #TBD
        
        # get node with lowest total cost
        openList.sort(key=lambda x: (heuristic(x[0]) + len(x[1])))
        current_board, path, depth = openList.pop(0)
        
        if depth > highest_depth:
            highest_depth = depth
        
        # check ig the goal is reached
        if current_board._board == current_board._target_board:
            return path, len(visited), i, highest_depth, round((time.time() - start)*1000, 3)
        
        state = tuple(tuple(i) for i in current_board._board)
        if state in visited:
            continue
        visited.add(state)
        
        # chceck neighbours
        for move in current_board.find_possible_moves('LRUD'):# order doesnt matter
            new_board = current_board.deepcopy()
            new_board.make_move(move)
            openList.append((new_board, path + [move], depth + 1))
            
        i+=1
        
    return -1, len(visited), i, highest_depth, round((time.time() - start)*1000, 3)
        
    
  
        

        
        
    
    


    

def test_bfs():
    b1 = brd.Board(4, 4)
    b1._board = [
    [1, 5, 9, 13],
    [2, 6, 10, 14],
    [4, 0, 7, 11],
    [8, 3, 12, 15]
    ]
    b1._blank_position = (3, 3)
    # b1._board = [
    # [1, 5, 9, 13],
    # [2, 6, 0, 14],
    # [3, 7, 10, 15],
    # [4, 8, 11, 12]
    # ]
    # b1._blank_position = (2, 2)
    # b1._board = [
    # [1, 5, 9, 13],
    # [2, 0, 11, 10],
    # [3, 6, 7, 14],
    # [4, 8, 12, 15]
    # ]
    
    # b1._blank_position = (1, 2)
    b1.display()
    
    print()
    solved = a_star(b1, 'hamm')
    print("długość znalezionego rozwiązania: ", len(solved[0]))
    print("liczba stanów odwiedzonych: ", solved[1])
    print("liczba stanów przetworzonych: ", solved[2])
    print("maksymalna głębokość: ", solved[3])
    print("czas trwania procesu obliczeniowego w milisekundach: ", solved[4])    
        

#lets test if it reaches all assumed scenarios            
def recurency_test(iter_number=0):
    if iter_number == 5:
        return 1
    lower_tiers_sum = 0
    for i in range (2):
        lower_tiers_sum += recurency_test(iter_number+1)
    return lower_tiers_sum

# print(recurency_test())
# test_bfs()
