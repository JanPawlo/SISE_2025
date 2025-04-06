import random

## board state managment in oop


class Board():
    
    def __init__(self, width, height):
        self._width = width
        self._height = height
        self._target_board = self._make_target_board()    
        self._board = list()
        self.shuffle_board()
        self._blank_position = self._find_blank()
        self._last_move = None

        
    def __eq__(self, other):
        return self._board == other._board

    def __hash__(self):
        return hash(self.board_to_tuple())

        
        
    def get_width(self):
        return self._width
    
    def get_height(self):
        return self._height

    def reset_blank_position(self):
        self._blank_position = self._find_blank()
        
    def _make_target_board(self):
        tablica = list()
        for i in range(self._width):
            tablica.append(list())
            for j in range(self._height):
                
                tablica[i].append((i+1)+j*self._width)
        ## pustka to zero
        tablica[i][j] = 0
        return tablica
    
    def shuffle_board(self):
        
        dostepne_liczby = list()
        for i in range (self._width*self._height):
            dostepne_liczby.append(i+1)
        
        ## pustka to zero
        dostepne_liczby[self._width*self._height-1] = 0
        random.shuffle(dostepne_liczby)
        # print(dostepne_liczby)
        

        for i in range(self._width):
            self._board.append(list())
            for j in range(self._height):
                
                self._board[i].append(dostepne_liczby[(i)+j*self._width])
                
    def _find_blank(self):
        for i in range(self._width):
            for j in range(self._height):
                if self._board[i][j] == 0:
                    return i, j
                
    def find_possible_moves(self):
        possible_moves = list()
    
        #todo: optimise if's
        if (self._blank_position[0] != 0):
            possible_moves.append('W')
        if (self._blank_position[0] != self._width-1):
            possible_moves.append('E')
        if (self._blank_position[1] != 0):
            possible_moves.append('N')
        if (self._blank_position[1] != self._height-1):
            possible_moves.append('S')
            
        return possible_moves
    
    def make_move(self, move):
        
        i, j = self._blank_position
        
        if (move == 'N'):
            self._board[i][j] = self._board[i][j-1]
            self._board[i][j-1] = 0
            self._last_move = move
            self._blank_position = (i, j-1)
        elif (move == 'E'):
            self._board[i][j] = self._board[i+1][j]
            self._board[i+1][j] = 0
            self._last_move = move
            self._blank_position = (i+1, j)
        elif (move == 'S'):
            self._board[i][j] = self._board[i][j+1]
            self._board[i][j+1] = 0
            self._last_move = move
            self._blank_position = (i, j+1)
        elif (move == 'W'):
            self._board[i][j] = self._board[i-1][j]
            self._board[i-1][j] = 0
            self._last_move = move
            self._blank_position = (i-1, j)
        else:
            raise ValueError("Wrong move format/name")
    
    def display(self):
        for i in range(self._height):
            for j in range(self._width):
                print(self._board[j][i], end=" ")
            print()
    
    
    # w zasadzie ta metoda narusza prywatnosc tablicy obiektu "copy"
    # moznaby odwrocic to, z ktorego obiektu jest wywolywanie
    def deepcopy(self):
        copy = Board(self._width, self._height)
        
        for i in range(self._width):
            for j in range(self._height):
                copy._board[i][j] = self._board[i][j]
        copy._blank_position = copy._find_blank()
        return copy
    
    def save_board(self):
        raise NotImplementedError()

    def load_board(self, path:str):
        raise NotImplementedError()
        


    
    
# tworzenie tablicy, tworzenie kopii, wykonywanie ruchow
def test_01():
    b1 = Board(4, 4)
    b1.display()
    b2 = b1.deepcopy()
    print('==')
    b1.make_move(b1.find_possible_moves()[0])
    b1.display()
    print()
    b2.display()
# test_01()
