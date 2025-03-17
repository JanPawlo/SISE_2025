import algorithms as alr
import board as brd
import sys


# def test_02():
#     width = 2
#     height = 2
#     b1 = brd.Board(width, height)
#     b1._board[0][0] = 1
#     b1._board[1][0] = 2
#     b1._board[0][1] = 0
#     b1._board[1][1] = 3
#     b1.reset_blank_position()
#     b1.display()
#     print()
#     wynik = alr.brute_force_solve(b1)    
#     if (isinstance(wynik, brd.Board)):
#         print()
#         wynik.display()


def test_03():
    counter = 0
    width = 2
    height = 2
    
    for i in range(50):
        b1 = brd.Board(width, height)
        # wynik = alr.bfs(b1)
        wynik = alr.brute_force_solve(b1)    
        if (isinstance(wynik, brd.Board)):
            counter += 1
    print ('sucess rate:', (counter/50)*100, '%')
    

#37
def main():

    # width = 2
    # height = 2
    # board = brd.Board(width, height)
    # print("Poczatkowa tablica:")
    # board.display()
    # print()
    
    # print("Wynik:")
    # wynik = alr.brute_force_solve(board)    
    # if (isinstance(wynik, brd.Board)):
    #     print("SUKCES")
    #     wynik.display()
    # test_02()
    test_03()    
    
main()




    # args = sys.argv
    # print(args)
    # #sraka, nie dziala
    # width = 2; height = 2;
    
    # # board = gnr.load_board()
    
    # board = gnr.generate_board(width, height)
    # target_board = gnr.generate_solved_board(width, height)

    # gnr.display_board(board)
    
    # alg.brute_force_solve(board, target_board)
    # # gnr.display_board(target_board)
    # # print(target_board)
    # print()
    # gnr.display_board(board)
    
    # # print(alg.find_blank(board))
    # # print(alg.find_possible_moves(board))
    # # alg.make_move(board, alg.find_possible_moves(board)[0])
    # # gnr.display_board(board)
    