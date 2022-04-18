class Tile:
    def __init__(self, position, is_bomb, number_of_close_bombs):
        self.position = position
        self.is_bomb = is_bomb
        self.number_of_close_bombs = number_of_close_bombs
        self.is_shown = False
        self.is_marked = False

    def init_bomb(position):
        return Tile(position, True, -1)

    def init_normal(position, number_of_close_bombs):
        return Tile(position, False, number_of_close_bombs)

    def show(self):
        self.is_shown = True
        self.is_marked = False

    def mark(self):
        if (self.is_shown):
            return
        
        self.is_marked = not self.is_marked
