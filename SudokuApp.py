import tkinter as tk
from tkinter import ttk, Menu, filedialog as fd, messagebox
import tkinter.font as tkFont
import numpy as np
from Sudoku import Sudoku
from Sudoku_Recognizer import Sudoku_Recognizer

class SudokuApp:
    GRID_SIZE = 9
    MIN_CLUES = 17
    EMPTY = 0
    COLOR_BG1 = '#19232d'
    COLOR_BG2 = '#33384d'
    COLOR_font1 = '#1781eb'
    COLOR_font2 = '#21de8c'
    
    def __init__(self, master):
        self.master = master
        self.master.title("Sudoku Solver v1.1.1")
        self.master.configure(bg=self.COLOR_BG1)
        self._center_window(600, 600)

        self.file_path = None
        self.cells = []
        self.font = tkFont.Font(size=14, weight='bold')
        self._setup_styles()
        self._create_menu()
        self._create_buttons()
        self._create_grid()
        self.master.attributes('-topmost', 1)
        
        self.sr = Sudoku_Recognizer()
        self.sr.load_digit_model()

    def _center_window(self, width, height):
        screen_width, screen_height = self.master.winfo_screenwidth(), self.master.winfo_screenheight()
        center_x = int(screen_width/2 - width/2)
        center_y = int(screen_height/2 - height/2)
        self.master.geometry(f'{width}x{height}+{center_x}+{center_y}')
        self.master.resizable(False, False)
        
    def _setup_styles(self):
        style = ttk.Style()
        style.theme_use('default')
        style.configure('Custom.TButton',
                        background=self.COLOR_BG2,
                        foreground=self.COLOR_font1,
                        font=('Arial', 12, 'bold'),
                        padding=6)
        style.map('Custom.TButton',
                  background=[('active', self.COLOR_BG2)],
                  foreground=[('active', self.COLOR_font1)])
        
    def _create_menu(self):
        menubar = Menu(self.master)
        self.master.config(menu=menubar)
        file_menu = Menu(menubar, tearoff=False)
        file_menu.add_command(label='New', accelerator='Ctrl+N', command=self.on_new)
        file_menu.add_command(label='Open...', accelerator='Ctrl+O', command=self.on_open)
        file_menu.add_command(label='Save', accelerator='Ctrl+S', command=self.on_save)
        file_menu.add_command(label='Save As...', accelerator='Ctrl+Shift+S', command=self.on_save_as)
        file_menu.add_separator()
        file_menu.add_command(label='Board Recognizer (BETA)', command=self.on_board_recognizer)
        file_menu.add_separator()
        file_menu.add_command(label='Exit', accelerator='Ctrl+Q', command=self.on_exit)
        menubar.add_cascade(label="File", menu=file_menu)
        self.master.config(menu=menubar)
        
        self.master.bind_all('<Control-n>', lambda e: self.on_new())
        self.master.bind_all('<Control-o>', lambda e: self.on_open())
        self.master.bind_all('<Control-s>', lambda e: self.on_save())
        self.master.bind_all('<Control-Shift-S>', lambda e: self.on_save_as())
        self.master.bind_all('<Control-q>', lambda e: self.on_exit())
        
    def _create_buttons(self):
        frame = tk.Frame(self.master, bg=self.COLOR_BG1)
        frame.pack(pady=10)
        
        solve_btn = ttk.Button(frame,
                                text='Solve',
                                style='Custom.TButton',
                                command=self.on_solve)
        solve_btn.pack(side='left', padx=10)
        
        clear_btn = ttk.Button(frame,
                                text='Clear',
                                style='Custom.TButton',
                                command=self.on_clear)
        clear_btn.pack(side='left', padx=10)
        
    def _create_grid(self):
        grid_frame = tk.Frame(self.master, bg=self.COLOR_BG1)
        grid_frame.pack(padx=40, pady=40)
        
        for r in range(self.GRID_SIZE):
            row_cells = []
            for c in range(self.GRID_SIZE):
                vcmd = (self.master.register(self.validate_input), '%P', r, c)
                cell = tk.Entry(grid_frame,
                                width=4,
                                justify='center',
                                font=self.font,
                                bg=self.COLOR_BG2  if (r//3 + c//3)%2 == 0 else self.COLOR_BG1,
                                validate='key',
                                validatecommand=vcmd)
                cell.grid(row=r, column=c, padx=2, pady=2, ipady=10)
                row_cells.append(cell)
            self.cells.append(row_cells)
    
    # Event Handlers:
        
    def on_solve(self):
        board_array = self.get_board_array()
        sudoku = Sudoku(board_array)
        
        valid, message = self.check_board_validity(sudoku)
        if not(valid):
            messagebox.showwarning("Sudoku Solver", message)
            return
        solved = sudoku.solve()
        
        if solved:
            self.set_board_array(sudoku)
            messagebox.showinfo("Sudoku Solver",
                                "The Sudoku board has been solved successfully")  
        else:
            messagebox.showwarning("Sudoku Solver", "The Sudoku board has no valid solution")
            
    def on_clear(self):
        for row in self.cells:
            for cell in row:
                cell.delete(0, tk.END)
                
    def on_new(self):
        self.file_path = None
        self.on_clear()
        
    def on_open(self):
        path = fd.askopenfilename(title='Open a file',                        
                                    filetypes=[('Text files', '*.txt')])
        
        if(not(path)):
            return
        
        try:
            with open(path, 'r') as file:
                data = file.read()
                data = ''.join(data.split())
                data = data.replace('.', '0')
                board0 = np.array([int(char) for char in data]).reshape((9, 9))
                board = Sudoku(board0)
            self.set_board_array(board)
        except:
            messagebox.showerror("Sudoku Solver", "Unable to load Sudoku board")
            
    def on_save(self):
        board = self.get_board_array()
        if self.file_path:
            np.savetxt(self.file_path, board, fmt='%d', delimiter='')
        else:
            self.on_save_as()
            
    def on_save_as(self):
        path = fd.asksaveasfilename(title='Save your file',
                                    defaultextension='.txt',
                                    filetypes=[('Text files', '*.txt')])
        if path:
            self.file_path = path
            np.savetxt(path, self.get_board_array(), fmt='%d', delimiter='')
    
    def on_board_recognizer(self):
        filepath = fd.askopenfilename(
                    filetypes=[("Select image file", "*.png;*.jpg;*.jpeg")])
        
        if not filepath:
            return
        
        self.sr.set_image_path(filepath)
        self.sr.read_image()
        self.sr.create_cell_image_list()
        self.sr.recognize_board()
        recognized_board = self.sr.get_recognized_board()
        board = Sudoku(recognized_board)
        self.set_board_array(board)
        
    def on_exit(self):
        self.master.destroy()
    
    # Helper functions:
        
    def get_board_array(self):
        grid = []
        for row in self.cells:
            vals = []
            for cell in row:
                text = cell.get()
                vals.append(int(text) if text.isdigit() else self.EMPTY)
            grid.append(vals)
        return np.array(grid)
    
    def set_board_array(self, sudoku):
        
        origin = sudoku.get_origin_board()
        current = sudoku.get_board()
        
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                val = current[r][c]
                txt = str(val) if val != self.EMPTY else ""
                cell = self.cells[r][c]
                cell.delete(0, tk.END)
                cell.insert(0, txt)
                cell.config(fg=self.COLOR_font2 if origin[r][c] == 0 and val != self.EMPTY
                                                else self.COLOR_font1)
                    
    def validate_input(self, new_val, r, c):
        
        r = int(r)
        c = int(c)
        
        if new_val.isdigit() and 1<=int(new_val)<=9 :
            self.cells[r][c].config(fg=self.COLOR_font1)
            return True
        if new_val == '':
            return True
        return False
                
    def check_board_validity(self, sudoku):
        # Check if board if valid before solving
        empty_cells = sudoku.get_empty_cells()
        
        if(empty_cells == self.GRID_SIZE**2):
            return False, "The Sudoku board is empty"
        elif(empty_cells > self.GRID_SIZE**2 - self.MIN_CLUES):
            return False, "More than one solution might exist (less than 17 clues)"
        
        if not(self.check_contradiction(sudoku)):
            return False, "Contradiction(s) exist in board"
        if empty_cells == 0:
            return False, "The Sudoku board is already solved"
        
        return True, "The board is valid and ready to solve"
    
    def check_contradiction(self, sudoku):
        board = sudoku.get_board()
        
        for i in range(self.GRID_SIZE):
            row_digits, col_digits = set(), set()
            for j in range(self.GRID_SIZE):
                # Rows check:
                val_row = board[i][j]
                if val_row != 0:
                    if val_row in row_digits:
                        return False
                    row_digits.add(val_row)
                
                # Columns check:
                val_col = board[j][i]
                if val_col != 0:
                    if val_col in col_digits:
                        return False
                    col_digits.add(val_col)
        # Blocks check:
        for b_ind in range(9):
            block_digits = set()
            for r, c in sudoku.scan_map['block'][b_ind]:
                val_block = sudoku.board[r][c]
                if(val_block != 0):
                    if(val_block in block_digits):
                        return False
                    block_digits.add(val_block)
        return True
 
if __name__ == '__main__':
    root = tk.Tk()
    app = SudokuApp(root)
    root.mainloop()