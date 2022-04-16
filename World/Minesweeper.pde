class Minesweeper {

  public boolean IsRunning = true;
  
  private Tile[][] _grid;
  private int _gridWidth = 0;
  private int _gridHeight = 0;
  private int _numberOfBombs = 0;
  
  private int _shownTiles = 0;
  
  private int[][] _helper = {
      { -1, -1 },
      { 0, -1 },
      { 1, -1 },
      { -1, 0 },
      { 1, 0 },
      { -1, 1 },
      { 0, 1 },
      { 1, 1 }
    };

  Minesweeper(int gridWidth, int gridHeight, int numberOfBombs) {
    _grid = new Tile[gridWidth][gridHeight];
    _gridWidth = gridWidth;
    _gridHeight = gridHeight;
    _numberOfBombs = numberOfBombs;

    int bombsPlaced = 0;

    while (bombsPlaced < numberOfBombs)
    {
      int rndX = int(random(0, gridWidth));
      int rndY = int(random(0, gridHeight));

      if (_grid[rndX][rndY] != null)
      {
        continue;
      }

      _grid[rndX][rndY] = new Tile(true, new PVector(rndX, rndY));
      bombsPlaced++;
    }

    for (int x = 0; x < gridWidth; x++)
    {
      for (int y = 0; y < gridHeight; y++)
      {
        if (_grid[x][y] != null)
        {
          continue;
        }

        int neighbouringBombs = 0;

        for (int i = 0; i < _helper.length; i++)
        {
          int tmpX = x + _helper[i][0];
          int tmpY = y + _helper[i][1];

          if (0 <= tmpX && tmpX < gridWidth
            && 0 <= tmpY && tmpY < gridHeight
            && _grid[tmpX][tmpY] != null && _grid[tmpX][tmpY].IsBomb)
          {
            neighbouringBombs++;
          }
        }

        _grid[x][y] = new Tile(neighbouringBombs, new PVector(x, y));
      }
    }
  }
  
  Tile GetTile(int x, int y)
  {
    return _grid[x][y];
  }
  
  Tile GetTile(PVector pos)
  {
    return _grid[int(pos.x)][int(pos.y)];
  }
  
  void ShowTile(Tile tile)
  {
    if (!IsRunning)
    {
      return;
    }
    
    ShowTile(tile, true);
  }
  
  private void ShowTile(Tile tile, boolean isStart)
  {
    if (tile.IsMarked)
    {
      return;
    }
    
    if (isStart && tile.IsShown && tile.NumberOfCloseBombs > 0)
    {
      int foundBombs = 0;
      
      for (int i = 0; i < _helper.length; i++)
      {
        int tmpX = int(tile.Position.x) + _helper[i][0];
        int tmpY = int(tile.Position.y) + _helper[i][1];

        if (0 <= tmpX && tmpX < gridWidth
          && 0 <= tmpY && tmpY < gridHeight
          && _grid[tmpX][tmpY].IsMarked)
        {
          foundBombs++;
        }
      }
      
      if (foundBombs == tile.NumberOfCloseBombs)
      {
        ShowNeighbouringTiles(tile);
      }
      return;
    }
    
    if (tile.IsShown)
    {
      return;
    }
    
    tile.Show();
    _shownTiles++;
    
    if (tile.IsBomb) 
    {
      IsRunning = false;
      ShowAllBombs();
      return;
    }
    
    if (_shownTiles == _gridWidth * _gridHeight - _numberOfBombs)
    {
      // win condition
      IsRunning = false;
      return;
    }
    
    if (tile.NumberOfCloseBombs == 0)
    {
      ShowNeighbouringTiles(tile);
    }
  }
  
  private void ShowNeighbouringTiles(Tile tile)
  {
    for (int i = 0; i < _helper.length; i++)
    {
      int tmpX = int(tile.Position.x) + _helper[i][0];
      int tmpY = int(tile.Position.y) + _helper[i][1];

      if (0 <= tmpX && tmpX < gridWidth
        && 0 <= tmpY && tmpY < gridHeight)
      {
        ShowTile(_grid[tmpX][tmpY], false);
      }
    } 
  }
  
  private void ShowAllBombs()
  {
    for (int x = 0; x < gridWidth; x++)
    {
      for (int y = 0; y < gridHeight; y++)
      {
        Tile tile = _grid[x][y];
        if (tile.IsBomb)
        {
          tile.Show();
        }
      }
    }
  }
  
  void MarkTile(Tile tile)
  {
    if (!IsRunning)
    {
      return;
    }
    
    if (tile.IsShown)
    {
      return;
    }
    
    tile.Mark();
  }

}
