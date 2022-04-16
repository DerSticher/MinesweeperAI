class Tile
{
  public boolean IsBomb = false;
  public int NumberOfCloseBombs = 0;
  
  public boolean IsShown = false;
  public boolean IsMarked = false;
  
  public PVector Position;

  Tile(boolean isBomb, PVector position)
  {
    IsBomb = isBomb;
    NumberOfCloseBombs = -1;
  }

  Tile(int numberOfCloseBombs, PVector position)
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
