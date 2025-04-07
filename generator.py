import random


def generate_board(width:int, height:int) -> list:
    
    dostepne_liczby = list()
    for i in range (width*height):
        dostepne_liczby.append(i+1)
    random.shuffle(dostepne_liczby)
    
    tablica = list()
    for i in range(width):
        tablica.append(list())
        for j in range(height):
            
            # tablica[i].append((i+1)+j*width)
            tablica[i].append(dostepne_liczby[(i)+j*width])
            
    return tablica

def generate_solved_board(width:int, height:int) -> list:
    tablica = list()
    for i in range(width):
        tablica.append(list())
        for j in range(height):
            
            tablica[i].append((i+1)+j*width)
            # tablica[i].append(dostepne_liczby[(i)+j*width])
            
    return tablica

def display_board(board:list):
    if len(board) == 0 or len(board[0])==0:
        raise ValueError("pusta plansza")
    for i in range(len(board)):
        for j in range(len(board[0])):
            print(board[j][i], end=" ")
        print()
        
        

def save_board(board:list):
    raise NotImplementedError()

def load_board(path:str) -> list:
    raise NotImplementedError()


# def shuffle_board(board:list):
    # np.random

# print(generate_board(4, 4))
     
# display_board(generate_board(4,4))