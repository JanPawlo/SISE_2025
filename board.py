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
                
    def find_possible_moves(self, order):
        
        possible_moves = list()
        
        i, j = self._blank_position
    
        #todo: optimise if's
        if (i != 0):
            possible_moves.append('W')
        if (i != self._width-1):
            possible_moves.append('E')
        if (j != 0):
            possible_moves.append('N')
        if (j != self._height-1):
            possible_moves.append('S')
            
        dir_map = {'L': 'W', 'R': 'E', 'U': 'N', 'D': 'S'}
        ordered = [dir_map[o] for o in order if dir_map[o] in possible_moves]
        return ordered
            

    
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

    def manhattan_heuristic(self):
        total_distance = 0
        
        for i in range(self._height):  # Loop over the rows
            for j in range(self._width):  # Loop over the columns
                tile_value = self._board[i][j]
                
                # Skip the blank tile (0)
                if tile_value == 0:
                    continue
                    
                # Calculate the target position of the current tile_value
                target_i, target_j = get_index(self._target_board, tile_value)
                

                # Calculate the Manhattan distance for the current tile
                total_distance += abs(i - target_i) + abs(j - target_j)
        
        return total_distance
    
    def hamming_heuristic(self):
        
        total_misplaced = 0
        
        for i in range(self._height):  # Loop over the rows
            for j in range(self._width):  # Loop over the columns
                tile_value = self._board[i][j]
                
                # Skip the blank tile (0)
                if tile_value == 0:
                    continue
                    
                # Calculate the target position of the current tile_value
                target_i, target_j = get_index(self._target_board, tile_value)
                
                # Check if the tile is in the wrong position
                if (i, j) != (target_i, target_j):
                    total_misplaced += 1
        
        return total_misplaced



def get_index(array, value):
    for i in range(len(array)):  # Iterate through the rows
        for j in range(len(array[i])):  # Iterate through the columns in each row
            if array[i][j] == value:
                return i, j  # Return the row and column indices (i, j)
    return None  # If the value is not found, return None



    
    
# tworzenie tablicy, tworzenie kopii, wykonywanie ruchow
def test_01():
    b1 = Board(4, 4)
    b1.display()
    # b2 = b1.deepcopy()
    print('==')
    b1.make_move(b1.find_possible_moves('LRUD')[0])
    b1.display()
    print()

    # b2.display()
# test_01()
