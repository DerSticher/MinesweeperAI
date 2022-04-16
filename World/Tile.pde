class Tile
{
  public boolean IsBomb = false;
  public int NumberOfCloseBombs = 0;
  
  public boolean IsShown = false;
  public boolean IsMarked = false;

  Tile(boolean isBomb)
  {
    IsBomb = isBomb;
    NumberOfCloseBombs = -1;
  }

  Tile(int numberOfCloseBombs)
  {
    IsBomb = false;
    NumberOfCloseBombs = numberOfCloseBombs;
  }
  
  void Show()
  {
    IsShown = true;
    IsMarked = false;
  }
  
  void Mark()
  {
    if (IsShown)
    {
      return;
    }
    
    IsMarked = !IsMarked;
  }
}
