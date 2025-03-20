import math
import pygame
import pickle

black = (0, 0, 0)
red = (255, 0, 0)
gray = (128, 128, 128)
green = (0, 255, 0)

REWARD_GOAL = 15
REWARD_PENALTY = -10


def save_positions(positions, filename):
    with open(filename, 'wb') as f:
        pickle.dump(positions, f)
        
def load_positions(filename):
    try:
        with open(filename, 'rb') as f:
            return pickle.load(f)
    except (FileNotFoundError, EOFError):
        
        return []  # Gibt eine leere Liste zurück, wenn die Datei nicht gefunden wird

class Car:
    def __init__(self, car_x, car_y, car_angle, car_speed, car_length, car_width, car_color, car_angle_speed):
        self.x = car_x
        self.y = car_y
        self.length = car_length
        self.width = car_width
        self.speed = car_speed
        self.angle_speed = car_angle_speed
        self.angle = car_angle
        self.color = car_color
        self.max_speed = 15
        
    def action(self, action):
        
        if action == 0: 
            self.angle += self.angle_speed
        elif action == 1:
            self.angle -= self.angle_speed
        elif action == 2:
            if self.speed < self.max_speed:
                self.speed += 1
            else: 
                self.speed = self.max_speed
        elif action == 3:
            if self.speed > 1: 
                self.speed -= 1
        
        self.x += self.speed * math.cos(math.radians(-self.angle))
        self.y += self.speed * math.sin(math.radians(-self.angle))
        

    
        
class RacingEnv:
    
    def __init__(self, size):
        pygame.init()
        self.screen = pygame.display.set_mode(size)
        self.walls = load_positions("real_walls")
        self.goals = load_positions("real_goals")
        self.goal_counter = 0
        self.fps = 60
        self.clock = pygame.time.Clock()
        self.game_reward = 0
        self.distances = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.max_d = 500
        self.font = pygame.font.Font(pygame.font.get_default_font(), 20)
        
    def get_length(self, x1, y1, x2, y2):
        return math.sqrt((x1-x2)**2 + (y1-y2)**2)
        
    def check_intersection(self, x3, y3, x4, y4, to_check):
    
        max_d = float("inf")
        closest = (0, 0)
        distance = 0
        
        for p in to_check:
        
            x1 = p[0][0]
            y1 = p[0][1]
            x2 = p[1][0]
            y2 = p[1][1]
        
                
            nenner = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        
            if nenner == 0:
                continue
            
            t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / nenner
            
            u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / nenner

            if t > 0 and t < 1 and u > 0 and u < 1:
                distance = self.get_length(x3, y3, x1 + t * (x2 - x1), y1 + t * (y2 - y1))
                if distance < max_d:
                    max_d = distance
                    closest = (x1 + t * (x2 - x1), y1 + t * (y2 - y1))
            else: 
                continue
            
        if closest:
            return closest, distance
        return (0, 0), 0
    
    def rotate_point(self, point, angle, center):
        """Dreht einen Punkt um einen Winkel relativ zu einem Mittelpunkt."""
        angle_rad = math.radians(angle)
        x, y = point
        cx, cy = center
        
        # Punkt relativ zum Mittelpunkt verschieben
        x -= cx
        y -= cy

        # Rotation anwenden (2D Rotationsmatrix)
        new_x = x * math.cos(angle_rad) - y * math.sin(angle_rad)
        new_y = x * math.sin(angle_rad) + y * math.cos(angle_rad)

        # Verschiebung zurück
        new_x += cx
        new_y += cy
        
        return new_x, new_y
    
    def draw_rotated_car_with_lines(self, car):
        # Auto zeichnen (ohne Rotation)
        rotated_car = pygame.Surface((car.width, car.length), pygame.SRCALPHA)
        pygame.draw.rect(rotated_car, car.color, (0, 0, car.width, car.length))
        
        # Auto rotieren
        rotated_car = pygame.transform.rotate(rotated_car, car.angle)
        
        # Rotiertes Rechteck erhalten
        rect = rotated_car.get_rect(center=(car.x, car.y))
            
    
        
        self.screen.blit(rotated_car, rect.topleft)
        return rotated_car
    
    def draw_rays(self, car, rotated_car):
        
        rect = rotated_car.get_rect(center=(car.x, car.y))
    
        point, distance = self.check_intersection(rect.center[0], rect.center[1], car.x + 1000*math.cos(math.radians(-car.angle)), car.y + 1000*math.sin(math.radians(-car.angle)), self.walls)
        pygame.draw.line(self.screen, gray, (rect.center), point)
        pygame.draw.circle(self.screen, red, point, 3)
        self.distances[0] = distance
        
        point, distance = self.check_intersection(rect.center[0], rect.center[1], car.x + 1000*math.cos(math.radians(-car.angle-90)), car.y + 1000*math.sin(math.radians(-car.angle-90)), self.walls)
        pygame.draw.line(self.screen, gray, (rect.center), point)
        pygame.draw.circle(self.screen, red, point, 3)
        self.distances[1] = distance
        
        point, distance = self.check_intersection(rect.center[0], rect.center[1], car.x + 1000*math.cos(math.radians(-car.angle+90)), car.y + 1000*math.sin(math.radians(-car.angle+90)), self.walls)
        pygame.draw.line(self.screen, gray, (rect.center), point)
        pygame.draw.circle(self.screen, red, point, 3)
        self.distances[2] = distance
        
        point, distance = self.check_intersection(rect.center[0], rect.center[1], car.x + 1000*math.cos(math.radians(-car.angle-45)), car.y + 1000*math.sin(math.radians(-car.angle-45)), self.walls)
        pygame.draw.line(self.screen, gray, (rect.center), point)
        pygame.draw.circle(self.screen, red, point, 3)
        self.distances[3] = distance
        
        point, distance = self.check_intersection(rect.center[0], rect.center[1], car.x + 1000*math.cos(math.radians(-car.angle+45)), car.y + 1000*math.sin(math.radians(-car.angle+45)), self.walls)
        pygame.draw.line(self.screen, gray, (rect.center), point)
        pygame.draw.circle(self.screen, red, point, 3)
        self.distances[4] = distance
        
        point, distance = self.check_intersection(rect.center[0], rect.center[1], car.x + 1000*math.cos(math.radians(-car.angle+22)), car.y + 1000*math.sin(math.radians(-car.angle+22)), self.walls)
        pygame.draw.line(self.screen, gray, (rect.center), point)
        pygame.draw.circle(self.screen, red, point, 3)
        self.distances[5] = distance
        
        point, distance = self.check_intersection (rect.center[0], rect.center[1], car.x + 1000*math.cos(math.radians(-car.angle-22)), car.y + 1000*math.sin(math.radians(-car.angle-22)), self.walls)
        pygame.draw.line(self.screen, gray, (rect.center), point)
        pygame.draw.circle(self.screen, red, point, 3)
        self.distances[6] = distance
        
        point, distance = self.check_intersection(rect.center[0], rect.center[1], car.x + 1000*math.cos(math.radians(-car.angle-67)), car.y + 1000*math.sin(math.radians(-car.angle-67)), self.walls)
        pygame.draw.line(self.screen, gray, (rect.center), point)
        pygame.draw.circle(self.screen, red, point, 3)
        self.distances[7] = distance
        
        point, distance = self.check_intersection(rect.center[0], rect.center[1], car.x + 1000*math.cos(math.radians(-car.angle+67)), car.y + 1000*math.sin(math.radians(-car.angle+67)), self.walls)
        pygame.draw.line(self.screen, gray, (rect.center), point)
        pygame.draw.circle(self.screen, red, point, 3)
        self.distances[8] = distance
        
    def check_walls(self, car):
        corners = [
            (car.x - car.width / 2, car.y - car.length / 2),  # oben links
            (car.x + car.width / 2, car.y - car.length / 2),  # oben rechts
            (car.x + car.width / 2, car.y + car.length / 2),  # unten rechts
            (car.x - car.width / 2, car.y + car.length / 2)   # unten links
        ]
        rotated_corners = [self.rotate_point(corner, -car.angle, (car.x, car.y)) for corner in corners]
        
        point, distance = self.check_intersection(rotated_corners[0][0], rotated_corners[0][1], rotated_corners[1][0], rotated_corners[1][1], self.walls)
        point2, distance2 = self.check_intersection(rotated_corners[2][0], rotated_corners[2][1], rotated_corners[3][0], rotated_corners[3][1], self.walls)
        point3, distance3 = self.check_intersection(rotated_corners[1][0], rotated_corners[1][1], rotated_corners[2][0], rotated_corners[2][1], self.walls)
        point4, distance4 = self.check_intersection(rotated_corners[3][0], rotated_corners[3][1], rotated_corners[0][0], rotated_corners[0][1], self.walls)
        
        if not (point == (0,0) and point2 == (0,0) and point3 == (0,0) and point4 == (0,0)):
            return True
        
        return False
              
    def check_goal(self, car):
        corners = [
            (car.x - car.width / 2, car.y - car.length / 2),  # oben links
            (car.x + car.width / 2, car.y - car.length / 2),  # oben rechts
            (car.x + car.width / 2, car.y + car.length / 2),  # unten rechts
            (car.x - car.width / 2, car.y + car.length / 2)   # unten links
        ]
        rotated_corners = [self.rotate_point(corner, -car.angle, (car.x, car.y)) for corner in corners]
        
        ugly_array = [self.goals[self.goal_counter], self.goals[self.goal_counter]]
        point2, distance2 = self.check_intersection(rotated_corners[2][0], rotated_corners[2][1], rotated_corners[3][0], rotated_corners[3][1], ugly_array)
        
        if not point2 == (0,0):
            if self.goal_counter < 27: 
                self.goal_counter += 1
                
            else:
                self.goal_counter = 0
            return True
        return False
        
    def draw(self, car): 
        
        
        self.clock.tick(self.fps)
        
        self.screen.fill((255, 255, 255))
        
        
        rotated_car = self.draw_rotated_car_with_lines(car)
        self.draw_rays(car, rotated_car)
        
        
        # draw goals
        count_to_next_goal = 0
        for p in self.goals: 
            x1 = p[0][0]
            y1 = p[0][1]
            x2 = p[1][0]
            y2 = p[1][1]
            
            if count_to_next_goal == self.goal_counter:
                pygame.draw.line(self.screen, red, (x1, y1), (x2, y2))
            else: 
                pygame.draw.line(self.screen, green, (x1, y1), (x2, y2))
            count_to_next_goal += 1
        #draw walls
        for p in self.walls:
            x1 = p[0][0]
            y1 = p[0][1]
            x2 = p[1][0]
            y2 = p[1][1]
            
            pygame.draw.line(self.screen, black, (x1, y1), (x2, y2))
        
        for i in pygame.event.get():
            
            if i.type == pygame.QUIT:
                return True
        
        text_surface = self.font.render(f'Speed: {car.speed}', True, pygame.Color('green'))
        self.screen.blit(text_surface, dest=(500, 25))
    
        pygame.display.update()
    
    def reset(self, car):
        car.x = 80
        car.y = 225
        self.goal_counter = 0
        car.angle = 90
        self.game_reward = 0
        car.speed = 1
    
    def step(self, car, action):
        observations = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        if action == -1:
            return False
        done = False
        car.action(action)
        
        
        reward = 0
        
        if self.check_goal(car):
            reward += REWARD_GOAL
        
        if self.check_walls(car):
            reward += REWARD_PENALTY
            done = True
        
        
            
        rotated_car = self.draw_rotated_car_with_lines(car)
        self.draw_rays(car, rotated_car)
        
        for i in range(len(self.distances)):
            observations[i] = self.distances[i] / self.max_d
        observations[9] = car.speed / car.max_speed
        return observations, reward, done
    
    def quit(self): 
        pygame.quit()