#testy wstepne
import board as brd
import algorithms as alr


def test_interactive():
    translate_dict = {'W':'N',
                      'D':'E',
                      'S':'S',
                      'A':'W'}
    
    width = 4
    height = 4
    b4 = brd.Board(width, height)
    print("Poczatkowa tablica:")
    b4.display()
    while(True):
        player_input = str(input())
        if player_input == 'P':
            break
        else:
            if player_input in translate_dict:
                player_input = translate_dict[player_input]    
                possible_moves = b4.find_possible_moves()
                if player_input not in set(possible_moves):
                    print("Wrong move dummy!")
                else:
                    b4.make_move(player_input)
                    b4.display()
            else:
                pass
            

test_interactive()