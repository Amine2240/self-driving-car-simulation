import os
import neat

import neat.config
import pygame
import math
import sys



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
    screen.blit(self.rotate_surface ,self.pos)
    self.draw_radar(screen)
  
  def draw_radar(self, screen):
    for radar in self.radars:
      pos , dist = radar
      pygame.draw.line(screen , (0 , 255 , 0) , self.center , pos , 1)
      pygame.draw.circle(screen, (0, 255, 0), pos, 5)
      
  
  def check_collision(self, map):
    self.is_alive = True
    for corner in self.corners: # corners of the car 
      if map.get_at((int(corner[0]) , int(corner[1]))) == (255 , 255 , 255 , 255):
        self.is_alive = False
        break
    
  
  def check_radar(self, degree , map): # calculate the distance of the sencor, its end-coordinates
    len = 0
    x = int(self.center[0])
    y = int(self.center[1])
    
    while not map.get_at((x, y)) == (255 , 255 , 255 , 255) and len < 300:
      len += 1
      x = int(self.center[0] + math.cos(math.radians(360 - (self.angle + degree)))*len) # degree is additional angle to cast the radar to specific direction relative to the car orientation
      y = int(self.center[1] + math.sin(math.radians(360 - (self.angle + degree)))*len)
    
    dist = int(math.sqrt((x - self.center[0])**2 + (y - self.center[1])**2))
    self.radars.append([(x,y) , dist])
  
  def update(self, map):
    self.speed =10
    
    self.rotate_surface = self.rot_center(self.surface , self.angle)
    
    
    self.pos[0] += math.cos(math.radians(360 -   self.angle))*self.speed 
    if self.pos[0] < 20:
      self.pos[0] = 20
    elif self.pos[0] > 1480: # 1500 - 20
      self.pos[0] = 1480
    
    
    #update the position of the car
    
    self.pos[1] += math.sin(math.radians(360 -  self.angle))*self.speed 
    if self.pos[1] < 20:
      self.pos[1] = 20
    elif self.pos[1] > 780: #800 -20
      self.pos[1] = 780
    
    
    
    self.center = [self.pos[0] + 50 , self.pos[1] + 50]
    len = 40
    #update the corners of the car
    left_top = [self.center[0] + math.cos(math.radians(360 - (self.angle + 30)))*len , self.center[1] + math.sin(math.radians(360 - (self.angle + 30)))*len ]
    right_top = [self.center[0] + math.cos(math.radians(360 - (self.angle + 150)))*len , self.center[1] + math.sin(math.radians(360 - (self.angle + 150)))*len ]
    left_bottom = [self.center[0] + math.cos(math.radians(360 - (self.angle + 210)))*len , self.center[1] + math.sin(math.radians(360 - (self.angle + 210)))*len ]
    right_bottom = [self.center[0] + math.cos(math.radians(360 - (self.angle + 330)))*len , self.center[1] + math.sin(math.radians(360 - (self.angle + 330)))*len ]
    
    
    
    self.corners = [left_top , right_top , left_bottom , right_bottom]
    
    
    self.distance += self.speed
    self.time_spent += 1  
    
    self.check_collision(map)
    
    self.radars.clear() # new radars at each position of the car
    for degree in range(-90 , 120 , 45): # ranges : -90 , -45 , 0 , 45 , 90 , 
      self.check_radar(degree , map)
  
  def get_data(self):
    radars_distances= [0,0,0,0,0]
    for i,item in enumerate(self.radars):
      radars_distances[i] = int(item[1]/30) # store the distance
    return radars_distances
  
  def get_alive(self):
    return self.is_alive
  
  def get_reward(self):
    return self.distance / 50
  
  def rot_center(self , image , angle): # when arriving to 'virage' rotate without affecting the position 
    orig_rect = image.get_rect()
    rot_image = pygame.transform.rotate(image , angle)
    rot_rect = orig_rect.copy()
    rot_rect.center = rot_image.get_rect().center
    rot_image = rot_image.subsurface(rot_rect).copy()
    return rot_image
  
  

generation = 0
def run_car(genomes , config):
  networks = []
  cars = []
  
  for  id, genome in genomes:
    net = neat.nn.FeedForwardNetwork.create(genome , config)
    networks.append(net)
    genome.fitness = 0
    cars.append(Car())
  
  pygame.init()
  screen = pygame.display.set_mode((1500, 800))
  clock = pygame.time.Clock()
  running = True
  map = pygame.image.load('./map.png')
  generation_font = pygame.font.SysFont("Arial", 70)
  font = pygame.font.SysFont('Arial' ,30)
  
  global generation
  generation+=1
  while running:
      # poll for events
      # pygame.QUIT event means the user clicked X to close your window
      for event in pygame.event.get():
          if event.type == pygame.QUIT:
              sys.exit(0)

      # RENDER YOUR GAME HERE
      
      
      
      
      for index, car in enumerate(cars):
        output = networks[index].activate(car.get_data())
        i = output.index(max(output))
        if i == 0:
          car.angle += 10
        else:
          car.angle -= 10
      
      remaincars = 0
      for i ,car in enumerate(cars):
        if car.get_alive():
          remaincars +=1
          car.update(map)
          genomes[i][1].fitness += car.get_reward()
      
      if remaincars == 0:
        break
      
      screen.blit(map , (0 , 0))
      
      for car in cars:
        if car.get_alive():
          car.draw(screen)
        
      generation_text_surface = generation_font.render("Generation: " + str(generation) , True ,(255, 255, 0))
      generation_text_rect = generation_text_surface.get_rect(center=(1500/2 , 100))
      
      remaincars_text_surface = font.render("remain cars: " + str(remaincars) , True ,(0, 0, 0))
      remaincars_text_rect = remaincars_text_surface.get_rect(center=(1500/2 , 200))
      
      
      
      screen.blit(generation_text_surface ,generation_text_rect )
      screen.blit(remaincars_text_surface , remaincars_text_rect)
      
      # flip() the display to put your work on screen
      
      pygame.display.flip()
      clock.tick(0)  # limits FPS to 60
  

  
  


if __name__ == "__main__":
  configfile = ('./config-feedforward.txt')
  config = neat.config.Config(neat.DefaultGenome , neat.DefaultReproduction ,neat.DefaultSpeciesSet , neat.DefaultStagnation , configfile)
  p = neat.Population(config)
  
  p.add_reporter(neat.StdOutReporter(True))
  stats =  neat.StatisticsReporter()
  p.add_reporter(stats)
  
  
  p.run(run_car , 1000)
  