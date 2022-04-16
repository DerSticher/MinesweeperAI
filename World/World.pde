
int tileSize = 25;

int gridWidth = 40;
int gridHeight = 20;
int numberOfBombs = 80;

Minesweeper game = new Minesweeper(gridWidth, gridHeight, numberOfBombs);
MinesweeperDrawer drawer = new MinesweeperDrawer(game, tileSize);
MinesweeperAI ai = new MinesweeperAI(game);

void settings()
{
  PVector dimensions = drawer.GetWindowSize();
  size(int(dimensions.x), int(dimensions.y));
}

void setup() {
  background(0);
  frameRate(2);
}

void draw()
{
  ai.PerformClick();
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
  
  Tile tile = game.GetTile(clickedTileX, clickedTileY);
  
  if (mouseButton == LEFT)
  {
    game.ShowTile(tile);
  }
  else if (mouseButton == RIGHT)
  {
    game.MarkTile(tile);
  }
}
