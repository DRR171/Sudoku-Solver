import numpy as np
from collections import defaultdict

class Sudoku:
    def __init__(self, board=None):
        if board is None:
            self.board = np.zeros((9, 9), dtype=int)
            self.board_origin = self.board.copy()
        else:
            self.board = board
            self.board_origin = board.copy()
        
        self.rows_list = np.zeros((9, 9), dtype=list)
        self.columns_list = np.zeros((9, 9), dtype=list)
        self.blocks_list = np.zeros((9, 9), dtype=list)
        
        for ind in range(9):
            self.rows_list[ind] = self._build_single_row_list(ind)
            self.columns_list[ind] = self._build_single_column_list(ind)
            self.blocks_list[ind] = self._build_single_block_list(ind)
        
        self.scan_map = {'row': self.rows_list,
                         'column': self.columns_list,
                         'block': self.blocks_list}    
        
        self.candidates = np.ones((9, 9, 9), dtype=bool)
        self._init_candidates()
        
        self.empty_cells = 81
        self._update_num_empty_cells()
        
    def get_board(self):
        return self.board
    
    def get_origin_board(self):
        return self.board_origin
    
    def get_candidates(self):
        return self.candidates
    
    def get_rows_list(self):
        return self.rows_list
    
    def get_columns_list(self):
        return self.columns_list
    
    def get_blocks_list(self):
        return self.blocks_list
    
    def get_empty_cells(self):
        return self.empty_cells
    
    def copy(self):
        new_sudoku = Sudoku(np.copy(self.board))
        new_sudoku.candidates = np.copy(self.candidates)
        new_sudoku.empty_cells = self.empty_cells
        new_sudoku.board_origin = self.board_origin.copy()
        
        return new_sudoku
        
    def _build_single_row_list(self, row):
        return [(row, col) for col in range(9)]
    
    def _build_single_column_list(self, col):
        return [(row, col) for row in range(9)]
    
    def _build_single_block_list(self, block_ind):
        # Generates a list of rows and columns to scan of the relevant block given a cell
        row_start = (block_ind // 3) * 3
        col_start = (block_ind % 3) * 3
        
        return [(r, c)
            for r in range(row_start, row_start + 3)
            for c in range(col_start, col_start + 3)]
    
    def _block_ind(self, row, col):
        # returns the block index of the given cell in position (row, col)
        return (row//3)*3 + (col//3)     
    
    def _set_cell_value(self, row, col, val, check_valid=False):
        #This function sets a value in a certain cell
        #needs to update candidates using update_candidates function
        if self.board[row][col] == 0:
            self.board[row][col] = val
            if not(check_valid) or self._is_safe_to_place(row, col, val):
                self._update_candidates(row, col)
                self.empty_cells -= 1
                return True
            else:
                self.board[row][col] = 0
                return False
        else:
            return False
    
    # Printing functions
    def print_board(self):
        print(self.board)
        
    def print_original_board(self):
        print(self.board_origin)
        
    def print_candidates(self):
        print(self.candidates)
    
    def _init_candidates(self):
    #initializes option matrix for a given board
        for r in range(9): #rows
            for c in range(9): #columns
                self._update_candidates(r, c)
            
    def _update_candidates(self, row, col):
    # This function updates (deletes) the candidates matrix
    # according to given row and column (specific cell)
    # If a cell has a digit, the candidates for this cell are all False
        if self.board[row][col] != 0:
            
            digit_index = self.board[row][col] - 1
            
            #eliminate all candidates in solved cell:
            for cand in range(9):
                self.candidates[row][col][cand] = False
        
            # Eliminate from column
            for r in range(9):
                self.candidates[r][col][digit_index] = False
            
            # Eliminate from row
            for c in range(9):
                self.candidates[row][c][digit_index] = False
            
            # Eliminate from block
            block_index = self._block_ind(row, col)
            for r, c in self.scan_map["block"][block_index]:    
                self.candidates[r][c][digit_index] = False
  
    def _update_num_empty_cells(self):
    # Updtes the number of empty cells in the board
        count = 0
        for r in range(9):
            for c in range(9):
                if self.board[r][c] == 0:
                    count += 1
        self.empty_cells = count

    def is_correct(self):
    # Checks if the current board is solved correctly
        def is_valid_group(group):
            return set(group) == set(range(1, 10))
    
        # Check rows
        for row in self.board:
            if not is_valid_group(row):
                return False
        
        # Check columns
        for col_idx in range(9):
            column = [self.board[row_idx][col_idx] for row_idx in range(9)]
            if not is_valid_group(column):
                return False
        
        # Check 3x3 blocks
        for block_row in range(0, 9, 3):
            for block_col in range(0, 9, 3):
                block = [self.board[r][c]
                    for r in range(block_row, block_row + 3)
                    for c in range(block_col, block_col + 3)]
                if not is_valid_group(block):
                    return False
        return True
             
    def is_valid(self):
    # checks of initial board is valid (at least 17 values, no contradictions)
        for r in range(9):
            for c in range(9):
                if not( 0 <= self.board[r][c] <= 9):
                    return False
        
        for i in range(9):
            row_digits = set()
            col_digits = set()
            for j in range(9):
                
                # Rows check:
                val_row = self.board[i][j]
                if val_row != 0:
                    if val_row in row_digits:
                        return False
                    row_digits.add(val_row)
                
                # Columns check:
                val_col = self.board[j][i]
                if val_col != 0:
                    if val_col in col_digits:
                        return False
                    col_digits.add(val_col)
        
        # Blocks check:
        for b in range(9):
            block_digits = set()
            for r, c in self.scan_map['block'][b]:
                val_block = self.board[r][c]
                if val_block != 0:
                    if val_block in block_digits:
                        return False
                    block_digits.add(val_block)
                        
        #Check if there are at least 17 digit inputs
        filled = sum(1 for r in range(9) for c in range(9) if self.board[r][c] != 0)
        if filled == 0:
            print("The Sudoku board is empty")
        if filled < 17:
            print("More than one solution might exist")
            
        return True
    
    def _is_safe_to_place(self, r, c, val):
    # Checks if setting val in row r and column c is valid
        # Check row:
        if val in self.board[r, :]:
            return False
        # Check column:
        if val in self.board[:, c]:
            return False
        # Check block:
        block_rows = (r//3)*3
        block_columns = (c//3)*3
        if val in self.board[block_rows:block_rows+3, block_columns:block_columns+3]:
            return False
        return True
    
    def board_status(self):
        empty_cells = self.get_empty_cells()
        if empty_cells == 81:
            return False, "The Sudoku board is empty"
        elif empty_cells > 64:
            return False, "More than one solution might exist (less than 17 clues)"
        elif not self.is_valid():
            return False, "Contradiction(s) exist in board"
        elif empty_cells == 0:
            return False, "The Sudoku board is already solved"
        return True, "The board is valid and ready to solve"

    
    def solve(self, verbose=False):
    # Solves the current Sudoku board
        if not(self.is_valid()):
            print("Sudoku board is not valid")
            return False
         
        while(self.empty_cells > 0):
            progress = 0
           
            # Simple Algorithms:
            progress += self._naked_single(verbose)
            progress += self._hidden_single(verbose)
           
            if progress > 0:
                continue
           
            # Advanced Algorithms:
            progress += self._naked_pair(verbose)
            progress += self._hidden_pair(verbose)
            progress += self._intersection_pointing(verbose)
            progress += self._intersection_claiming(verbose)
            progress += self._X_wing_rows(verbose)
            progress += self._X_wing_columns(verbose)
            
            if progress > 0:
                continue
            
            progress += self._backtracking(verbose)
            
            if progress == 0:
                if verbose:
                    print("No further progress can be made with the current technics.\n")
                break
        return self.is_correct()
             
    #Simple algorithms:
    def _naked_single(self, verbose=False):
        count = 0
        cnd_found = None
        total_found = 0
        
        for r in range(9): #rows
            for c in range(9): #columns
                if self.board[r][c] == 0:
                    for cand in range(9): #candidates
                        if self.candidates[r][c][cand]:
                            count += 1
                            cnd_found = cand
                    if count == 1: #Naked single found
                        self._set_cell_value(r, c, cnd_found+1, False)
                        total_found += 1
                        if verbose:
                            print(f"Found naked single digit: {cnd_found+1} at ({r},{c})\n")
                    count = 0
                    cnd_found = None
        return total_found

    def _hidden_single(self, verbose=False):
        total_found = 0
        for scan_type, scan_list in self.scan_map.items():
            
            for cnd in range(9):
                for i in range(9):
                    count = 0
                    found_cell = []
                    for j in range(9):
                        row, col = scan_list[i][j]
                        if self.candidates[row][col][cnd]:
                            count += 1
                            found_cell = [row, col]
                    if count == 1 and found_cell:
                        self._set_cell_value(found_cell[0], found_cell[1], cnd+1, False)
                        total_found += 1
                        if verbose:
                            print(f"Found hidden single digit: {cnd+1} at "
                                  f"({found_cell[0]},{found_cell[1]}) in {scan_type}\n")
        return total_found
    
    # Advanced algorithms:
    # These algorithms help eliminate candidates from the board to allow the simple algorithms
    # to solve more cells
    def _naked_pair(self, verbose=False):
        eliminated = 0
        for scan_type, scan_list in self.scan_map.items():
            for i in range(9):
                pair_dict = defaultdict(list)
                for j in range(9):
                    row, col = scan_list[i][j]
                    if self.board[row][col] != 0:
                        continue
                    cell_cand = self.candidates[row][col]
                    if sum(cell_cand) == 2:
                        digits = frozenset(ind + 1 for ind, val in enumerate(cell_cand) if val)
                        pair_dict[digits].append((row, col))
                
                for digits, positions in pair_dict.items():
                    if len(positions) == 2:
                        eliminate_list = scan_list[i]
                        for r, c in eliminate_list:
                            if (r, c) in positions:
                                continue
                            if self.board[r][c] != 0:
                                continue
                            
                            cell_cand = self.candidates[r][c]
                            for d in digits:
                                if cell_cand[d-1]:
                                    cell_cand[d-1] = False
                                    eliminated += 1
                                    if verbose:
                                        print(f"Removed {d} candidate from cell ({r}, {c})"
                                              f" due to naked pair {sorted(digits)} at {positions} in {scan_type}\n")
        return eliminated
            
    def _hidden_pair(self, verbose=False):
        eliminated = 0
        for scan_type, scan_list in self.scan_map.items():
            for i in range(9):
                digit_pos = defaultdict(list)
                for d in range(9):
                    unit_cells = scan_list[i]

                    positions = [(r, c) for r, c in unit_cells
                                 if self.board[r][c] == 0 and self.candidates[r, c, d]] 
                    
                    if len(positions) == 2:
                        digit_pos[d] = positions
                if len(digit_pos) < 2:
                    continue
                
                keys = list(digit_pos.keys())
                for k in range(len(keys)):
                    for j in range(k + 1, len(keys)):
                        d1 = keys[k]
                        d2 = keys[j]
                        if set(digit_pos[d1]) == set(digit_pos[d2]):
                            for r, c in digit_pos[d1]:
                                for elim_d in range(9):
                                    if elim_d != d1 and elim_d != d2 and self.candidates[r, c, elim_d]:
                                        self.candidates[r, c, elim_d] = False
                                        eliminated += 1
                                        if verbose:
                                            print(f"Removed candidate {elim_d+1} from ({r}, {c}) "
                                                  f"due to hidden pair {d1+1}, {d2+1} in {scan_type} {i}\n")
        return eliminated
                                    
    # Intersections
    def _intersection_pointing(self, verbose=False):
        # Block scanning and row/column elimination
        eliminated = 0
        for block_ind in range(9):
            # block_cells = self.scan_map['block'][block_ind]
            block_cells = [tuple(cell) for cell in self.scan_map['block'][block_ind]]
            for d in range(9):
                digit_pos = [(r, c) for r, c in block_cells
                             if self.board[r][c] == 0 and self.candidates[r, c, d]]
                
                if 2 <= len(digit_pos) <= 3:
                    rows = {r for r, _ in digit_pos}
                    cols = {c for _, c in digit_pos}
                    
                    if len(rows) == 1:
                        # scan other columns (outside the block) and eliminate the digit
                        row = rows.pop()
                        for c in range(9):
                            if self.board[row][c] == 0 and (row, c) not in block_cells:
                                if self.candidates[row, c, d]:
                                    self.candidates[row, c, d] = False
                                    eliminated += 1
                                    if verbose:
                                        print(f"Removed candidate {d+1} from ({row}, {c}) "
                                              f"due to intersection (pointing) in block {block_ind}\n")
                    if len(cols) == 1:
                        # scan other rows (outside the block) and eliminat the digit
                        col = cols.pop()
                        for r in range(9):
                            if self.board[r][col] == 0 and (r, col) not in block_cells:
                                if self.candidates[r, col, d]:
                                    self.candidates[r, col, d] = False
                                    eliminated += 1
                                    if verbose:
                                        print(f"Removed candidate {d+1} from ({r}, {col}) "
                                              f"due to intersection (pointing) in block {block_ind}\n")
                    
        return eliminated
                    
    def _intersection_claiming(self, verbose=False):
        # Row/Column scanning and block elimination
        eliminated = 0
        for ind in range(9):
            #scan through blocks
            row_cells = [tuple(cell) for cell in self.scan_map['row'][ind]]
            column_cells = [tuple(cell) for cell in self.scan_map['column'][ind]]
            
            for d in range(9):
                digit_pos_row = [(r, c) for r, c in row_cells
                             if self.board[r][c] == 0 and self.candidates[r, c, d]]
                
                if 2 <= len(digit_pos_row) <= 3:
                    block_ind_set = {self._block_ind(r, c) for r, c in digit_pos_row}
                    
                    if len(block_ind_set) == 1:
                        # scan in block (outside the row) and eliminate the digit
                        block_ind = block_ind_set.pop()
                        block_cells = [tuple(cell) for cell in self.scan_map['block'][block_ind]]
                        for r, c in block_cells:
                            if r != ind and (r, c) not in digit_pos_row and self.board[r][c] == 0:
                                if self.candidates[r, c, d]:
                                    self.candidates[r, c, d] = False
                                    eliminated += 1
                                    if verbose:
                                        print(f"Removed candidate {d+1} from ({r}, {c}) "
                                              f"due to intersection (claiming) in row {ind}\n")
                
                digit_pos_column = [(r, c) for r, c in column_cells
                             if self.board[r][c] == 0 and self.candidates[r, c, d]]    
                if 2 <= len(digit_pos_column) <= 3:
                    block_ind_set = {self._block_ind(r, c) for r, c in digit_pos_column}
                    
                    if len(block_ind_set) == 1:
                        # scan in block (outside the column) and eliminat the digit
                        block_ind = block_ind_set.pop()
                        block_cells = [tuple(cell) for cell in self.scan_map['block'][block_ind]]
                        for r, c in block_cells:
                            if c != ind and (r, c) not in digit_pos_column and self.board[r][c] == 0:
                                if self.candidates[r, c, d]:
                                    self.candidates[r, c, d] = False
                                    eliminated += 1
                                    if verbose:
                                        print(f"Removed candidate {d+1} from ({r}, {c}) "
                                              f"due to intersection (pointing) in column {ind}\n")
        return eliminated            
    
    def _X_wing_rows(self, verbose=False):
        eliminated = 0
        for d in range(9):
            for r1 in range(9):
                cols_r1 = [c for c in range(9) if self.candidates[r1, c, d]]
                if len(cols_r1) != 2:
                    continue
                for r2 in range(r1 + 1, 9):                
                    cols_r2 = [c for c in range(9) if self.candidates[r2, c, d]]
                    if cols_r2 != cols_r1:
                        continue
                    c1, c2 = cols_r1
                    for elim_r in range(9):
                        if elim_r in (r1, r2):
                            continue
                        for c in (c1, c2):
                            if self.board[elim_r][c] == 0 and self.candidates[elim_r, c, d]:
                                self.candidates[elim_r, c, d] = False
                                eliminated += 1
                                if verbose:
                                    print(f"Removed candidate {d+1} from ({elim_r}, {c}) "
                                          f"due to X-wing on columns {c1}, {c2} in rows {r1}, {r2}\n")
        return eliminated
                
    def _X_wing_columns(self, verbose=False):
        eliminated = 0
        for d in range(9):
            for c1 in range(9):
                rows_c1 = [r for r in range(9) if self.candidates[r, c1, d]]
                if len(rows_c1) != 2:
                    continue
                for c2 in range(c1 + 1, 9):                
                    rows_c2 = [r for r in range(9) if self.candidates[r, c2, d]]
                    if rows_c2 != rows_c1:
                        continue
                    r1, r2 = rows_c1
                    for elim_c in range(9):
                        if elim_c in (c1, c2) :
                            continue
                        for r in (r1, r2):
                            if self.board[r][elim_c] == 0 and self.candidates[r, elim_c, d]:
                                self.candidates[r, elim_c, d] = False
                                eliminated += 1
                                if verbose:
                                    print(f"Removed candidate {d+1} from ({r}, {elim_c}) "
                                          f"due to X-wing on rows {r1}, {r2} in columns {c1}, {c2}\n")
        return eliminated
    
    # Backtracking recursive algorithm in case the conventional algorithms don't work
    def _backtracking(self, verbose=False):
        temp_sudoku = self.copy()
        return self._backtracking_support(temp_sudoku, verbose)
        
    def _backtracking_support(self, temp_sudoku, verbose):
        if temp_sudoku.is_correct():
            self.board = np.copy(temp_sudoku.board)
            self.candidates = np.copy(temp_sudoku.candidates)
            self.empty_cells = temp_sudoku.empty_cells
            return True
        
        cells = [(r, c, sum(temp_sudoku.candidates[r, c]))
                for r in range(9)
                for c in range(9)
                if temp_sudoku.board[r][c] == 0 and sum(temp_sudoku.candidates[r, c]) > 1]
        
        if not cells:
            return False
        
        r, c, _ = min(cells, key=lambda x: x[2])
        
        candidates = np.where(temp_sudoku.candidates[r, c])[0]
        
        for cand in candidates:
            guess_val = cand + 1
            guess_sudoku = temp_sudoku.copy()
            if verbose:
                print(f"Backtracking: Trying cell ({r}, {c}) with candidate {cand+1}\n")
            guess_sudoku._set_cell_value(r, c, guess_val, False)
            guess_sudoku.solve(verbose)
            
            if guess_sudoku.is_correct():
                self.board = np.copy(guess_sudoku.board)
                self.candidates = np.copy(guess_sudoku.candidates)
                self.empty_cells = guess_sudoku.empty_cells
                return True
        return False