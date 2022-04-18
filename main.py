from Minesweeper import Minesweeper
from MinesweeperDrawer import MinesweeperDrawer
from MinesweeperUserInput import MinesweeperUserInput

tile_size = 25

width = 40
height = 20
number_of_bombs = 80

game = Minesweeper(width, height, number_of_bombs)
drawer = MinesweeperDrawer(game, tile_size)
input = MinesweeperUserInput(game, tile_size)

while 1:
    input.handle_input()
    drawer.draw()