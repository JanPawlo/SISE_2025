import board as brd



def bfs(board, order, max_iter=10000):
    visited = set()
    queue = [(board.deepcopy(), [])]
    i = 0

    while queue and i < max_iter:
        current_board, path = queue.pop(0)
        state = current_board.board_to_tuple()

        if state in visited:
            continue

        visited.add(state)

        if current_board._board == current_board._target_board:
            print(f"Solution found in {len(path)} moves after {i} iterations")
            print("Move sequence:", path)
            current_board.display()
            return path  # Optionally return the final board too

        for move in current_board.find_possible_moves():
            new_board = current_board.deepcopy()
            new_board.make_move(move)
            queue.append((new_board, path + [move]))

        i += 1

        if i % 1000 == 0:
            print(f"Iteration: {i}, Queue size: {len(queue)}")
        if i % 10000 == 0:
            current_board.display()

    print("No solution found within iteration limit.")
    return None


def dfs(board, order, max_iter=10000):
    
    visited = set()
    stack = [(board.deepcopy())]  
    i = 0
    
    while i < max_iter and stack:
        
        current_board = stack.pop()
                
        if current_board not in visited:
            
            visited.add(current_board)
    
            if current_board._board == current_board._target_board:
                print(f"Solution found in {i} moves")
                return current_board
    
            for move in current_board.find_possible_moves():
                new_board = current_board.deepcopy()
                new_board.make_move(move)
                stack.append(new_board)

        i += 1
        if i % 1000 == 0:
            print(f"Iteration: {i}, Stack size: {len(stack)}")
        if i % 10000 == 0:
            current_board.display()

    print("No solution found within iteration limit.")
    return None

    

def test_bfs():
    b1 = brd.Board(4, 4)
    b1.display()
    
    print()
    solved = bfs(b1, None, 100000000)
    if solved is not None:
        print(solved)
    else:
        print("Nie rozwiazano")
    

#lets test if it reaches all assumed scenarios            
def recurency_test(iter_number=0):
    if iter_number == 5:
        return 1
    lower_tiers_sum = 0
    for i in range (2):
        lower_tiers_sum += recurency_test(iter_number+1)
    return lower_tiers_sum

# print(recurency_test())
test_bfs()
