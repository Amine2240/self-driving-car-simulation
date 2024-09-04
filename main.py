import os
import neat

import neat.config
import neat.population
import pygame



class Car:
  def __init__(self):   
    self.surface = pygame.image.load("car.png")
    self.surface = pygame.transform.scale(self.surface, (100, 100))
    self.rotate_surface = self.surface
    self.pos = [700, 650]
    self.angle = 0
    self.speed = 0
    self.center = [self.pos[0] + 50, self.pos[1] + 50]
    self.radars = []
    self.radars_for_draw = []
    self.is_alive = True
    self.goal = False
    self.distance = 0
    self.time_spent = 0
    
  def draw(self, screen):
    pass
  
  def draw_radar(self, screen):
    pass
  
  def check_collision(self, screen):
    pass
  
  def check_radar(self, screen):
    pass
  
  def update(self, screen):
    pass
  
  def get_data(self, screen):
    pass
  
  def get_alive(self, screen):
    pass
  
  def get_reward(self, screen):
    pass
  
  def rot_center(self, screen):
    pass
  
  

def run_car(genomes ,config):
  networks = []
  cars = []
  
  for genome in genomes:
    networks.append(neat.nn.FeedForwardNetwork.create(genome , config))
    genome.fitness = 0
    cars.append(Car())
  
  
  pygame.init()
  screen = pygame.display.set_mode((1500, 800))
  clock = pygame.time.Clock()
  running = True
  map = pygame.image.load('./map.png')
  generation_font = pygame.font.SysFont("Arial", 70)
  font = pygame.font.SysFont('Arial' ,30)
  # mycar = pygame.image.load("./car.png")
  
  while running:
      # poll for events
      # pygame.QUIT event means the user clicked X to close your window
      for event in pygame.event.get():
          if event.type == pygame.QUIT:
              running = False

      # RENDER YOUR GAME HERE
      
      screen.blit(map ,(0,0))
      
      
      # flip() the display to put your work on screen
      
      pygame.display.flip()

      clock.tick(0)  # limits FPS to 60

  pygame.quit()
  
  


if __name__ == "__main__":
  
  config_path = './config-feedforward.text'
  config = neat.config.Config(neat.DefaultGenome , neat.DefaultReproduction , neat.DefaultSpeciesSet , neat.DefaultStagnation , config_path)
  
  
  p = neat.Population(config)
  
  p.add_reporter(neat.StdOutReporter(True))
  stats = neat.StatisticsReporter()
  p.add_reporter(stats)
  
  p.run(run_car ,1000)
  