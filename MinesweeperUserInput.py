import pygame, sys

class MinesweeperUserInput:
    def __init__(self, game, tile_size):
        self.game = game
        self.tile_size = tile_size

    def handle_input(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT: sys.exit()
            elif event.type == pygame.MOUSEBUTTONUP:
                pos = (int(event.pos[0] / self.tile_size), int(event.pos[1] / self.tile_size))
                tile = self.game.get_tile(pos)

                if (event.button == pygame.BUTTON_LEFT):
                    self.game.show_tile(tile)
                elif (event.button == pygame.BUTTON_RIGHT):
                    self.game.mark_tile(tile)
