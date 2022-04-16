class MinesweeperDrawer
{
  private Minesweeper _game;
  private int _tileSize;
  
  MinesweeperDrawer(Minesweeper game, int tileSize)
  {
    _game = game;
    _tileSize = tileSize;
  }
  
  PVector GetWindowSize()
  {
    return new PVector(_game._gridWidth * _tileSize, _game._gridHeight * _tileSize);
  }
  
  void Draw()
  {
    background(0);
    
    for (int x = 0; x < _game._gridWidth; x++)
    {
      for (int y = 0; y < _game._gridHeight; y++)
      {
        DrawTile(x, y);
      }
    }
  }
  
  void DrawTile(int xPos, int yPos)
  {
    Tile tile = _game._grid[xPos][yPos];
    
    int x = xPos * _tileSize + 1;
    int y = yPos * _tileSize + 1;
    int size = _tileSize - 2;
    
    if (tile.IsMarked)
    {
      fill(0, 0, 192);
      rect(x, y, size, size);
      return;
    }
    
    if (!tile.IsShown)
    {
      fill(128);
      rect(x, y, size, size);
      return;
    }

    if (tile.IsBomb)
    {
      fill(255, 0, 0);
      rect(x, y, size, size);
    } 
    else
    {
      fill(192);
      rect(x, y, size, size);

      fill(0);
      textAlign(CENTER, CENTER);
      textSize(size * 0.8);
      text(str(tile.NumberOfCloseBombs), x + size / 2 + 1, y + size / 2 - 3);
    }
  }
}
