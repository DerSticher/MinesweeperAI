class MinesweeperAI
{
  private Minesweeper _game;
  
  MinesweeperAI(Minesweeper game)
  {
    _game = game;
  }
  
  void PerformClick()
  {
    // mark bomb we know of
    for (int x = 0; x < _game._gridWidth; x++)
    {
      for (int y = 0; y < _game._gridHeight; y++)
      {
        Tile tile = _game._grid[x][y];
        if (!tile.IsShown)
        {
          continue;
        }

        ArrayList<Tile> neighbours = GetNeighbours(x, y);
        neighbours.removeIf(n -> n.IsShown);
        int tmpSize = neighbours.size();
        neighbours.removeIf(n -> n.IsMarked);
        int alreadyMarked = tmpSize - neighbours.size();
        
        if (neighbours.size() + alreadyMarked == tile.NumberOfCloseBombs)
        {
          neighbours.forEach(n -> _game.MarkTile(n));
        }
      }
    }
    
    // click on tile where we already found all bombs
    for (int x = 0; x < _game._gridWidth; x++)
    {
      for (int y = 0; y < _game._gridHeight; y++)
      {
        Tile tile = _game._grid[x][y];
        if (!tile.IsShown)
        {
          continue;
        }

        ArrayList<Tile> neighbours = GetNeighbours(x, y);
        neighbours.removeIf(n -> !n.IsMarked);
        
        if (neighbours.size() == tile.NumberOfCloseBombs)
        {
          _game.ShowTile(tile);
        }
      }
    }
    
    // fallback click on any unshown tile
    Tile rndTile;
    do
    {
      int rndX = int(random(0, _game._gridWidth));
      int rndY = int(random(0, _game._gridHeight));
      
      rndTile = _game._grid[rndX][rndY]; 
    } while(rndTile.IsShown);
    
    _game.ShowTile(rndTile);
  }
  
  private ArrayList<Tile> GetNeighbours(int x, int y)
  {
    ArrayList<Tile> neighbours = new ArrayList<Tile>();
    for (int i = 0; i < _game._helper.length; i++)
    {
      int tmpX = x + _game._helper[i][0];
      int tmpY = y + _game._helper[i][1];

      if (0 <= tmpX && tmpX < gridWidth
        && 0 <= tmpY && tmpY < gridHeight)
      {
        neighbours.add(_game._grid[tmpX][tmpY]);
      }
    }
    return neighbours;
  }
}
