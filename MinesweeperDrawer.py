from numpy import array
import pygame

class MinesweeperDrawer:
    def __init__(self, game, tile_size):
        self.game = game
        self.tile_size = tile_size

        pygame.init()
        self.screen = pygame.display.set_mode((self.game.width * self.tile_size, self.game.height * self.tile_size))

        self.font = pygame.font.SysFont('Arial', int(tile_size * 0.8))
        self.numbers = []
        for i in range(9):
            self.numbers.append(self.font.render(str(i), True, (0, 0, 0)))

    def draw(self):
        self.screen.fill((0, 0, 0))
    
        for x in range(self.game.width):
            for y in range(self.game.height):
                self.draw_tile(x, y)

        pygame.display.flip()
  
    def draw_tile(self, x_pos, y_pos):
        tile = self.game.grid[x_pos][y_pos]
    
        x = x_pos * self.tile_size + 1
        y = y_pos * self.tile_size + 1
        size = self.tile_size - 2
    
        if (tile.is_marked):
            pygame.draw.rect(self.screen, (0, 0, 192), (x, y, size, size))
            return
    
        if (not tile.is_shown):
            pygame.draw.rect(self.screen, (128, 128, 128), (x, y, size, size))
            return

        if (tile.is_bomb):
            pygame.draw.rect(self.screen, (255, 0, 0), (x, y, size, size))
        else:
            pygame.draw.rect(self.screen, (192, 192, 192), (x, y, size, size))

            number_surface = self.numbers[tile.number_of_close_bombs]
            self.screen.blit(number_surface, (x + size / 4 + 1, y))
            # text(str(tile.NumberOfCloseBombs), x + size / 2 + 1, y + size / 2 - 3)
