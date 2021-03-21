import pygame
import neat
import time
import os
import random
import pickle

pygame.font.init()

WIN_WIDTH = 500
WIN_HEIGHT = 800
BIRD_IMGS = [pygame.transform.scale2x(pygame.image.load(os.path.join('imgs', 'bird1.png'))),
             pygame.transform.scale2x(pygame.image.load(os.path.join('imgs', 'bird2.png'))),
             pygame.transform.scale2x(pygame.image.load(os.path.join('imgs', 'bird3.png')))]
PIPE_IMG = pygame.transform.scale2x(pygame.image.load(os.path.join('imgs', 'pipe.png')))
BASE_IMG = pygame.transform.scale2x(pygame.image.load(os.path.join('imgs', 'base.png')))
BG_IMG = pygame.transform.scale2x(pygame.image.load(os.path.join('imgs', 'bg.png')))

STAT_FONT = pygame.font.SysFont('comicsans', 50)
GEN = 0


class Bird:
    IMGS = BIRD_IMGS
    MAX_ROTATION = 25
    ROT_VEL = 20
    ANIMATION_TIME = 5

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.tilt = 0
        self.tick_count = 0
        self.vel = 0
        self.height = self.y
        self.img_count = 0
        self.img = self.IMGS[0]

    def jump(self):
        self.vel = -10.5
        self.tick_count = 0
        self.height = self.y

    def move(self):
        self.tick_count += 1
        d = self.vel * self.tick_count + 1.5 * self.tick_count ** 2

        if d >= 16:
            d = 16
        elif d < 0:
            d -= 2

        self.y = self.y + d

        if d < 0 or self.y < self.height + 50:
            if self.tilt < self.MAX_ROTATION:
                self.tilt = self.MAX_ROTATION
        else:
            if self.tilt > -90:
                self.tilt -= self.ROT_VEL

    def draw(self, win):
        self.img_count += 1

        if self.img_count < self.ANIMATION_TIME:
            self.img = self.IMGS[0]
        elif self.img_count < self.ANIMATION_TIME * 2:
            self.img = self.IMGS[1]
        elif self.img_count < self.ANIMATION_TIME * 3:
            self.img = self.IMGS[2]
        elif self.img_count < self.ANIMATION_TIME * 4:
            self.img = self.IMGS[1]
        elif self.img_count == self.ANIMATION_TIME * 4 + 1:
            self.img = self.IMGS[0]
            self.img_count = 0

        if self.tilt <= -80:
            self.img = self.IMGS[1]
            self.img_count = self.ANIMATION_TIME * 2

        rotated_image = pygame.transform.rotate(self.img, self.tilt)
        new_rect = rotated_image.get_rect(center=self.img.get_rect(topleft=(self.x, self.y)).center)
        win.blit(rotated_image, new_rect.topleft)

    def get_mask(self):
        return pygame.mask.from_surface(self.img)


class Pipe:
    GAP = 200
    VEL = 5

    def __init__(self, x):
        self.x = x
        self.height = 0
        self.top = 0
        self.bottom = 0
        self.PIPE_TOP = pygame.transform.flip(PIPE_IMG, False, True)
        self.PIPE_BOTTOM = PIPE_IMG

        self.passed = False
        self.set_height()

    def set_height(self):
        self.height = random.randrange(50, 450)
        self.top = self.height - self.PIPE_TOP.get_height()
        self.bottom = self.height + self.GAP

    def move(self):
        self.x -= self.VEL

    def draw(self, win):
        win.blit(self.PIPE_TOP, (self.x, self.top))
        win.blit(self.PIPE_BOTTOM, (self.x, self.bottom))

    def collide(self, bird):
        bird_mask = bird.get_mask()
        top_mask = pygame.mask.from_surface(self.PIPE_TOP)
        bottom_mask = pygame.mask.from_surface(self.PIPE_BOTTOM)

        top_offset = (self.x - bird.x, self.top - round(bird.y))
        bottom_offset = (self.x - bird.x, self.bottom - round(bird.y))

        b_point = bird_mask.overlap(bottom_mask, bottom_offset)
        t_point = bird_mask.overlap(top_mask, top_offset)

        if b_point or t_point:
            return True
        return False


class Base:
    VEL = 5
    WIDTH = BASE_IMG.get_width()
    IMG = BASE_IMG

    def __init__(self, y):
        self.y = y
        self.x1 = 0
        self.x2 = self.WIDTH

    def move(self):
        self.x1 -= self.VEL
        self.x2 -= self.VEL

        if self.x1 + self.WIDTH <= 0:
            self.x1 = self.x2 + self.WIDTH

        if self.x2 + self.WIDTH <= 0:
            self.x2 = self.x1 + self.WIDTH

    def draw(self, win):
        win.blit(self.IMG, (self.x1, self.y))
        win.blit(self.IMG, (self.x2, self.y))


def draw_window(win, birds, pipes, base, score, gen):
    win.blit(BG_IMG, (0, 0))

    for pipe in pipes:
        pipe.draw(win)

    base.draw(win)

    for bird in birds:
        bird.draw(win)

    text = STAT_FONT.render('Score: ' + str(score), 1, (255, 255, 255))
    win.blit(text, (WIN_WIDTH - 10 - text.get_width(), 10))

    text = STAT_FONT.render('Gen: ' + str(gen), 1, (255, 255, 255))
    win.blit(text, (10, 10))

    pygame.display.update()


def main(genomes, config):
    global GEN
    GEN += 1
    nets = []
    ge = []
    # bird = Bird(230, 350)
    birds = []

    for _, g in genomes:
        net = neat.nn.FeedForwardNetwork.create(g, config)  # setting up a neural network for each of our genomes
        nets.append(net)  # list of all our genomes
        birds.append(Bird(230, 350))  # creating the bird that corresponds to its neural network
        g.fitness = 0  # setting the start fitness to 0
        ge.append(g)

    base = Base(730)
    pipes = [Pipe(600)]
    win = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
    clock = pygame.time.Clock()

    score = 0

    run = True
    while run:
        # so we can see progress in real time, but if we want to train agents we don't need to slow it down and draw anything to the screen
        clock.tick(30)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                pygame.quit()
                quit()

        pipe_ind = 0
        if len(birds) > 0:
            if len(pipes) > 1 and birds[0].x > pipes[0].x + pipes[0].PIPE_TOP.get_width():
                pipe_ind = 1
        else:
            run = False
            break

        for x, bird in enumerate(birds):
            bird.move()
            ge[x].fitness += 0.1  # giving the bird reward for staying alive and moving forward

            # passing all the inputs to neural network
            output = nets[x].activate((bird.y, abs(bird.y - pipes[pipe_ind].height),
                                       abs(bird.y - pipes[pipe_ind].bottom)))

            # making decisions based on the neural networks output
            if output[0] > 0.5:  # btw output is list, but we generate 1 by 1
                bird.jump()

        add_pipe = False
        rem = []
        for pipe in pipes:
            for x, bird in enumerate(birds):
                if pipe.collide(bird):
                    ge[x].fitness -= 1  # every time bird hits the pipe we will decrease fitness (give negative reward)
                    # if ge[x].fitness < 0:
                    #     ge[x].fitness = 0
                    birds.pop(x)
                    nets.pop(x)  # when the bird dies we will stop progression of its fitness
                    ge.pop(x)

                if not pipe.passed and pipe.x <= bird.x:
                    pipe.passed = True
                    add_pipe = True

            if pipe.x + pipe.PIPE_TOP.get_width() < 0:
                rem.append(pipe)

            pipe.move()

        if add_pipe:
            score += 1
            pipes.append(Pipe(600))
            for g in ge:
                # all birds that have survived to this point get +5 reward
                g.fitness += 5

        for r in rem:
            pipes.remove(r)

        for x, bird in enumerate(birds):
            if bird.y + bird.img.get_width() >= 730 or bird.y < 0:
                ge[x].fitness -= 1
                birds.pop(x)
                nets.pop(x)  # when the bird dies we will stop progression of its fitness
                ge.pop(x)

        if score > 100:
            break  # so we can stop very good agents and get back our winner object

        base.move()
        draw_window(win, birds, pipes, base, score, GEN)


# !!! NEAT START !!! #

def run(config_path):
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)

    p = neat.Population(config)  # creating the population

    # For creating reports about generations (not required)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    # Set the fitness function that we will run for 50 generations (fitness, generations) + we need to change our main function (add genomes and config parametars)
    winner = p.run(main, 5)  # we can save our agent + neural network by saving this winner object using pickle
    pickle_out = open('bird.pickle', 'wb')
    pickle.dump(winner, pickle_out)
    pickle_out.close()


def replay_genome(config_path, genome_path="bird.pickle"):
    # Load required NEAT config
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)

    # Unpickle saved winner
    with open(genome_path, "rb") as f:
        genome = pickle.load(f)

    # Convert loaded genome into required data structure
    genomes = [(1, genome)]

    # Call game with only the loaded genome
    main(genomes, config)


if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config-feedforward.txt")
    replay_genome(config_path)

# !!! NEAT END !!! #


# Explanation ++

# 1. inputs
# 2. outputs (actions)
# 3. activation function (just for input, neat decides for hidden layers)
# 4. population size (neat only)
# 5. fitness function
# 6. max generations

# [NEAT]
# fitness_criterion     = max (min/mean) -- how to choose the best agent
# fitness_threshold     = 100 -- what is the fitness goal for our agent
# pop_size              = 20 -- population size
# reset_on_extinction   = True -- do we want to delete bad species
#
# [DefaultGenome] - genomes are connections between nodes
# # node activation options
# activation_default      = tanh -- activation function
# activation_mutate_rate  = 0.0 -- % chance that new activation function will be created for next generation
# activation_options      = tanh -- array from which mutate rate will choose from (space separated)
#
# # node aggregation options
# aggregation_default     = sum
# aggregation_mutate_rate = 0.0
# aggregation_options     = sum -- (space separated) one of sum, product, min, max, mean, median, and maxabs
#
# # node bias options
# bias_init_mean          = 0.0 -- bias options
# bias_init_stdev         = 1.0
# bias_max_value          = 30.0
# bias_min_value          = -30.0
# bias_mutate_power       = 0.5
# bias_mutate_rate        = 0.7
# bias_replace_rate       = 0.1
#
# # genome compatibility options
# compatibility_disjoint_coefficient = 1.0
# compatibility_weight_coefficient   = 0.5
#
# # connection add/remove rates
# conn_add_prob           = 0.5 -- probability for adding new connections
# conn_delete_prob        = 0.5 -- probability for deleting connections
#
# # connection enable options
# enabled_default         = True
# enabled_mutate_rate     = 0.01
#
# feed_forward            = True
# initial_connection      = full -- Dense neurons
#
# # node add/remove rates
# node_add_prob           = 0.2 -- probability for adding new nodes
# node_delete_prob        = 0.2 -- probability for deleting new nodes
#
# # network parameters
# num_hidden              = 0 -- start number for hidden neurons !
# num_inputs              = 3 -- default number of input neurons !
# num_outputs             = 1 -- default number of output neurons !
#
# # node response options
# response_init_mean      = 1.0
# response_init_stdev     = 0.0
# response_max_value      = 30.0
# response_min_value      = -30.0
# response_mutate_power   = 0.0
# response_mutate_rate    = 0.0
# response_replace_rate   = 0.0
#
# # connection weight options
# weight_init_mean        = 0.0
# weight_init_stdev       = 1.0
# weight_max_value        = 30 -- weights limits
# weight_min_value        = -30
# weight_mutate_power     = 0.5
# weight_mutate_rate      = 0.8
# weight_replace_rate     = 0.1
#
# [DefaultSpeciesSet]
# compatibility_threshold = 3.0
#
# [DefaultStagnation]
# species_fitness_func = max
# max_stagnation       = 20 -- if this number of generations passes and our fitness stays the same we will kill that species
# species_elitism      = 2 -- number of species immune to max_stagnation extinction
#
# [DefaultReproduction]
# elitism            = 2
# survival_threshold = 0.2
