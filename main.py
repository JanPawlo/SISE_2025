import sys
from algorithms import bfs, dfs, a_star
from board import Board



def save_solution(path, moves):
    f = open(path, 'w')
    
    if moves == -1:
        f.write("-1")
    else:
        f.write(f"{len(moves)}\n")
        f.write("".join(moves))

def save_stats(path, movesNum, visited, processed, depth, time_ms):
    f = open(path, 'w')
    
    f.write(f"{movesNum}\n")
    f.write(f"{processed}\n")
    f.write(f"{visited}\n")
    f.write(f"{depth}\n")
    f.write(f"{time_ms}\n")
    
    f.close()

if __name__ == "__main__":
    if len(sys.argv) != 6:
        print("Usage: python main.py <strategy> <param> <input_file> <solution_file> <stats_file>")
        sys.exit(1)

    strategy, param, input_file, sol_file, stats_file = sys.argv[1:]

    board = Board(1, 1) # height and width here dont matter
    board.load_board(input_file)

    if strategy == "bfs":
        moves, visited, processed, depth, time_ms = bfs(board, param)
    elif strategy == "dfs":
        moves, visited, processed, depth, time_ms = dfs(board, param)
    elif strategy == "astr":
        moves, visited, processed, depth, time_ms = a_star(board, param)
    else:
        print(f"Unknown strategy: {strategy}")
        sys.exit(2)

    save_solution(sol_file, moves)
    save_stats(stats_file, len(moves), visited, processed, depth, time_ms)

    