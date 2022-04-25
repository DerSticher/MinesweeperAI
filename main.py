from statistics import mean
from Minesweeper import Minesweeper
from MinesweeperDrawer import MinesweeperDrawer
from MinesweeperUserInput import MinesweeperUserInput
from Agent import Agent
import matplotlib.pyplot as plt
import pygame
import time

use_ai = True

tile_size = 25

width = 9
height = 9
number_of_bombs = 10

game = Minesweeper(width, height, number_of_bombs)
drawer = MinesweeperDrawer(game, tile_size)

def plot(scores, mean_scores):
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    plt.plot(scores)
    plt.plot(mean_scores)
    plt.ylim(ymin=0)
    plt.text(len(scores) - 1, scores[-1], str(scores[-1]))
    plt.text(len(mean_scores) - 1, mean_scores[-1], str(mean_scores[-1]))
    plt.pause(0.01)



def perform_action(action):
    # print('Perform Action', action)

    type = action[0]
    x = action[1]
    y = action[2]
    shown_tiles_before = game.shown_tiles

    if not ((0 <= x < game.width) and (0 <= y < game.height)):
        return (-10, not game.is_running, game.shown_tiles)

    tile = game.get_tile((x, y))
    reward_mod = 0

    if (type == 0):
        game.show_tile(tile)

        if tile.is_bomb:
            reward_mod -= 10
        elif not game.is_running:
            reward_mod += 10
    elif (type == 1):
        game.mark_tile(tile)

        if not tile.is_shown:
            if not tile.is_marked:
                if tile.is_bomb:
                    reward_mod += 0.5
                else:
                    reward_mod -= 0.5
            else:
                if tile.is_bomb:
                    reward_mod -= 0.5
                else:
                    reward_mod += 0.5
    else:
        reward_mod -= 1

    reward = (game.shown_tiles - shown_tiles_before) * 2
    if reward == 0:
        reward = -10
    
    return (reward + reward_mod, not game.is_running, game.shown_tiles)


plot([0], [0])
# time.sleep(3)

if (use_ai):
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent(game)

    clock = pygame.time.Clock()
    clock.tick()

    elapsed_time = 0

    while 1:
        pygame.event.get()

        elapsed_time += clock.get_time()

        if (elapsed_time > 0):
            elapsed_time = 0
            state_old = agent.get_state()

            final_move = agent.get_action(state_old)

            reward, done, score = perform_action(final_move)
            # print(reward, done, score)
            state_new = agent.get_state()
            agent.train_short_memory(state_old, final_move, reward, state_new, done)

            agent.remember(state_old, final_move, reward, state_new, done)

            drawer.draw()

            if done:
                print('epsilon', agent.epsilon)
                # time.sleep(0.2)
                game.reset()
                agent.n_games += 1
                agent.train_long_memory()

                if score > record:
                    record = score
                    agent.model.save()

                plot_scores.append(score)
                total_score += score
                mean_score = total_score / agent.n_games
                plot_mean_scores.append(mean_score)
                plot(plot_scores, plot_mean_scores)
                print('Game', agent.n_games, 'Score', score, 'Record', record, 'Mean Score', mean_score)

        clock.tick()

else:

    input = MinesweeperUserInput(game, tile_size)

    while 1:
        input.handle_input()
        drawer.draw()
