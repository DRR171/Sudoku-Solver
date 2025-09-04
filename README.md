![Sudoku Logo](./logo/Sudoku_Solver_logo.png)
# Sudoku-Solver
Sudoku Solver is a personal project written in Python.
This application lets users:
- create Sudoku boards manually
- save and load boards from a file
- import a board from an image using a CNN-based digit recognizer  
It includes a GUI for interaction and a solving engine that can handle puzzles of varying difficulty.

---

## Installation

### Clone the repository
```bash
git clone https://github.com/DRR171/SudokuSolver.git
cd SudokuSolver

python -m venv .venv
# Activate it:
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
