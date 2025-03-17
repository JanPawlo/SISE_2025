import board as brd



def brute_force_solve(board, prev_move='X', iter_number=0):
    
    
    if board._board == board._target_board:
        return board
    #wykonaj kopię
    #zrob na niej ruch
    # ^ te dwa to wlasciwie ten sam ruch
    
    #sprawdz poprawnosc
    #zatrzymaj jezeli poprawne ALBO osiagnieto maks glebo
    #wywolaj kolejna rekurencje (najlepiej wzdluz galezi)
    
    if iter_number >= 100:
        pass
    else:
        if prev_move == 'N':
            block = 'S'
        elif prev_move == 'S':
            block = 'N'
        elif prev_move == 'W':
            block = 'E'
        elif prev_move == 'E':
            block = 'W'
        elif prev_move == 'X':
            block = "X"
        else:
            raise ValueError("Wrong previous move value")
    
        possible_moves = board.find_possible_moves()
        for move in possible_moves:
            if not (move == block):                
                cp_b = board.deepcopy()
                cp_b.make_move(move)
                # print('==', iter_number)
                # cp_b.display()
                # if cp_b._board == board._target_board:
                #     return cp_b
                result = brute_force_solve(cp_b, move, iter_number+1)
                if isinstance(result, brd.Board):
                    if iter_number >= 10:
                        print(iter_number)
                    return result
                
                
                
                
                
def bfs(board, iter_number=1):
    
    if board._board == board._target_board:
        return board
    
    if iter_number >= 20:
        return False
    
    else:
        possible_moves = board.find_possible_moves()
        for move in possible_moves:
            cp_b = board.deepcopy()
            cp_b.make_move(possible_moves[0])
            result = bfs(cp_b, iter_number+1)
            if result != False:
                return result



#lets test if it reaches all assumed scenarios            
def recurency_test(iter_number=0):
    if iter_number == 5:
        return 1
    lower_tiers_sum = 0
    for i in range (2):
        lower_tiers_sum += recurency_test(iter_number+1)
    return lower_tiers_sum

# print(recurency_test())
