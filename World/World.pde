
int tileSize = 25;

int gridWidth = 40;
int gridHeight = 20;
int numberOfBombs = 80;

Minesweeper game = new Minesweeper(gridWidth, gridHeight, numberOfBombs);
MinesweeperDrawer drawer = new MinesweeperDrawer(game, tileSize);

void settings()
{
  PVector dimensions = drawer.GetWindowSize();
  size(int(dimensions.x), int(dimensions.y));
}

void setup() {
  background(0);
}

void draw()
{
  drawer.Draw();
}

void mouseReleased()
{
  if (!game.IsRunning)
  {
   return; 
  }
  
  int clickedTileX = int(mouseX / tileSize);
  int clickedTileY = int(mouseY / tileSize);
  
  if (mouseButton == LEFT)
  {
    game.ShowTile(clickedTileX, clickedTileY);
  }
  else if (mouseButton == RIGHT)
  {
    game.MarkTile(clickedTileX, clickedTileY);
  }
}
