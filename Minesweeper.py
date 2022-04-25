from random import randint
from Tile import Tile
import gc

class Minesweeper:

    helper = [[-1, -1], [0, -1], [1, -1], [-1, 0], [1, 0], [-1, 1], [0, 1], [1, 1]]

    def __init__(self, width, height, number_of_bombs):
        self.width = width
        self.height = height
        self.number_of_bombs = number_of_bombs
        self.reset()
    
    def reset(self):
        if hasattr(self, 'grid'):
            del self.grid
            gc.collect()

        self.grid = []
        for x in range(self.width):
            self.grid.append([0] * self.height)

        self.shown_tiles = 0
        bombs_placed = 0

        while (bombs_placed < self.number_of_bombs):
            rnd_x = randint(0, self.width - 1)
            rnd_y = randint(0, self.height - 1)

            if (self.grid[rnd_x][rnd_y] != 0):
                continue
                
            self.grid[rnd_x][rnd_y] = Tile.init_bomb((rnd_y, rnd_y))
            bombs_placed += 1

        for x in range(self.width):
            for y in range(self.height):
                if (self.grid[x][y] != 0):
                    continue

                close_bombs = 0
                helper = Minesweeper.helper

                for i in range(len(helper)):
                    tmp_x = x + helper[i][0]
                    tmp_y = y + helper[i][1]

                    if (0 <= tmp_x < self.width and 0 <= tmp_y < self.height and self.grid[tmp_x][tmp_y] != 0 and self.grid[tmp_x][tmp_y].is_bomb):
                        close_bombs += 1
                    
                self.grid[x][y] = Tile.init_normal((x, y), close_bombs)

        self.is_running = True

  

    def get_tile(self, position):
        return self.grid[position[0]][position[1]]
  
    def show_tile(self, tile):
        if (not self.is_running):
            return
        
        self.show_tile_internal(tile, True)
  
    def show_tile_internal(self, tile, is_start):
        if (tile.is_marked):
            return
    
        if (is_start and tile.is_shown and tile.number_of_close_bombs > 0):
            found_bombs = 0
      
            helper = Minesweeper.helper

            for i in range(len(helper)):
                tmp_x = tile.position[0] + helper[i][0]
                tmp_y = tile.position[1] + helper[i][1]

                if (0 <= tmp_x < self.width and 0 <= tmp_y < self.height and self.grid[tmp_x][tmp_y].is_marked):
                    found_bombs += 1
                
            if (found_bombs == tile.number_of_close_bombs):
                self.show_neighbouring_tiles(tile)
            
            return
    
        if (tile.is_shown):
            return
    
        tile.show()
        self.shown_tiles += 1
    
        if (tile.is_bomb): 
            self.is_running = False
            self.show_all_bombs()
            return
    
        if (self.shown_tiles == self.width * self.height - self.number_of_bombs):
            # win condition
            self.is_running = False
            return
        
        if (tile.number_of_close_bombs == 0):
            self.show_neighbouring_tiles(tile)
  
    def show_neighbouring_tiles(self, tile):
        helper = Minesweeper.helper

        for i in range(len(helper)):
            tmp_x = tile.position[0] + helper[i][0]
            tmp_y = tile.position[1] + helper[i][1]

            if (0 <= tmp_x < self.width and 0 <= tmp_y < self.height):
                self.show_tile_internal(self.grid[tmp_x][tmp_y], False)

  
    def show_all_bombs(self):
        for x in range(self.width):
            for y in range(self.height):
                tile = self.grid[x][y]
                if (tile.is_bomb):
                    tile.show()


    def mark_tile(self, tile):
        if (not self.is_running):
            return
    
        if (tile.is_shown):
            return
        
        tile.mark()
