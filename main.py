import pygame
import pymunk
import pymunk.pygame_util
import numpy as np
import math
import config
from game_env import Bike, Terrain
from neural_net import NeuralNetwork
import sys

def main():
    pygame.init()
    screen = pygame.display.set_mode((config.SCREEN_WIDTH, config.SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    draw_options = pymunk.pygame_util.DrawOptions(screen)

    # Genetic Algorithm Variables
    generation = 0
    population = []
    
    # Initialize Population
    input_size = 5 + config.RAY_COUNT
    hidden_size = 8
    output_size = 2 # Gas/Brake, Lean
    
    for _ in range(config.POPULATION_SIZE):
        nn = NeuralNetwork(input_size, hidden_size, output_size)
        population.append({'nn': nn, 'fitness': 0})

    while True:
        # Setup Simulation for this generation
        space = pymunk.Space()
        space.gravity = config.GRAVITY
        
        terrain = Terrain(space)
        
        bikes = []
        for ind in population:
            bike = Bike(space, (100, 300))
            bikes.append({'bike': bike, 'nn': ind['nn'], 'active': True})
            
        frame_count = 0
        max_frames = 60 * 30 # 30 seconds max per gen
        
        simulating = True
        
        while simulating:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    sys.exit()
            
            # Physics Step
            dt = 1.0 / config.FPS
            space.step(dt)
            frame_count += 1
            
            # Update Bikes
            alive_count = 0
            for obj in bikes:
                if not obj['active']:
                    continue
                    
                bike = obj['bike']
                if not bike.alive:
                    obj['active'] = False
                    continue
                
                alive_count += 1
                
                # Get inputs
                state = bike.get_state()
                
                # Predict
                action = obj['nn'].predict(state)
                
                # Apply action
                bike.update(action)
                
            # Render
            screen.fill((255, 255, 255))
            
            # Follow the best bike (camera)
            # Find furthest active bike
            best_bike = None
            max_x = -float('inf')
            for obj in bikes:
                if obj['active'] and obj['bike'].chassis_body.position.x > max_x:
                    max_x = obj['bike'].chassis_body.position.x
                    best_bike = obj['bike']
            
            # If all dead or timeout
            if alive_count == 0 or frame_count > max_frames:
                simulating = False
                break
                
            # Camera transform
            if best_bike:
                camera_x = -best_bike.chassis_body.position.x + config.SCREEN_WIDTH / 3
                # Draw Terrain
                # We need to manually draw or use pymunk debug draw with transform
                # Pymunk pygame util doesn't easily support camera offset without hack
                # So we just shift everything for drawing? No, too slow.
                # Let's just use the debug draw and accept it might be static or we hack the surface.
                
                # Simple custom drawing
                # Draw Terrain
                for shape in terrain.shapes:
                    p1 = shape.a
                    p2 = shape.b
                    start_pos = (int(p1.x + camera_x), int(config.SCREEN_HEIGHT - p1.y))
                    end_pos = (int(p2.x + camera_x), int(config.SCREEN_HEIGHT - p2.y))
                    pygame.draw.line(screen, (0, 0, 0), start_pos, end_pos, 5)
                
                # Draw Bikes
                for obj in bikes:
                    if not obj['active']: continue
                    b = obj['bike']
                    
                    # Chassis
                    p = b.chassis_body.position
                    pos = (int(p.x + camera_x), int(config.SCREEN_HEIGHT - p.y))
                    # Rotate? Pymunk bodies rotate.
                    # Simplified drawing for now: Circles for wheels, Rect for body
                    # To draw rotated rect in pygame is annoying, let's just draw lines/circles
                    
                    # Wheels
                    for w_body in [b.rear_wheel_body, b.front_wheel_body]:
                        wp = w_body.position
                        w_pos = (int(wp.x + camera_x), int(config.SCREEN_HEIGHT - wp.y))
                        pygame.draw.circle(screen, (50, 50, 50), w_pos, config.WHEEL_RADIUS)
                        # Spoke to see rotation
                        angle = w_body.angle
                        end_spoke = (w_pos[0] + math.cos(angle)*config.WHEEL_RADIUS, w_pos[1] - math.sin(angle)*config.WHEEL_RADIUS)
                        pygame.draw.line(screen, (200, 200, 200), w_pos, end_spoke, 2)

                    # Chassis (Line)
                    c_pos = b.chassis_body.position
                    c_angle = b.chassis_body.angle
                    # Draw a line representing chassis
                    # Ideally use vertices
                    # Get world vertices
                    poly = b.chassis_shape
                    # Pymunk 6+ logic
                    # We can iterate vertices
                    # For visualization speed let's just draw a line from rear to front wheel
                    pygame.draw.line(screen, (200, 0, 0), 
                        (int(b.rear_wheel_body.position.x + camera_x), int(config.SCREEN_HEIGHT - b.rear_wheel_body.position.y)),
                        (int(b.front_wheel_body.position.x + camera_x), int(config.SCREEN_HEIGHT - b.front_wheel_body.position.y)),
                        5)
                        
                    # Draw Rays
                    if hasattr(b, 'last_rays'):
                         for start, end in b.last_rays:
                             s = (int(start[0] + camera_x), int(config.SCREEN_HEIGHT - start[1]))
                             e = (int(end[0] + camera_x), int(config.SCREEN_HEIGHT - end[1]))
                             pygame.draw.line(screen, (0, 255, 0), s, e, 1)

            # Info text
            font = pygame.font.SysFont("Arial", 18)
            text = font.render(f"Gen: {generation} | Alive: {alive_count}", True, (0, 0, 0))
            screen.blit(text, (10, 10))
            
            pygame.display.flip()
            clock.tick(config.FPS)

        # End of generation
        # Calculate fitness
        fitnesses = []
        for obj in bikes:
            # Fitness = distance
            fit = obj['bike'].distance_traveled
            # Bonus for staying alive?
            # fit += obj['bike'].time_alive * 0.1
            fitnesses.append(fit)
            # Update the persistent population object
            # Finding the matching neural net
            # Actually we just rebuilt 'bikes' list from 'population' list in order
            # So indices match.
        
        # Assign fitness back to population
        for i, val in enumerate(fitnesses):
            population[i]['fitness'] = val
            
        # Logging
        best_fit = max(fitnesses)
        print(f"Generation {generation} complete. Best Fitness: {best_fit:.2f}")
        
        # Selection & Crossover
        # Sort by fitness desc
        population.sort(key=lambda x: x['fitness'], reverse=True)
        
        new_population = []
        
        # Elitism
        for i in range(config.ELITISM_COUNT):
            new_population.append(population[i]) # Keep exact copy? Or copy logic?
            # Better to copy the NN to avoid reference issues if we mutate later
            new_ind = {'nn': population[i]['nn'].copy(), 'fitness': 0}
            new_population.append(new_ind)

        # Fill rest
        while len(new_population) < config.POPULATION_SIZE:
            # Tournament selection
            # Pick 2 random
            p1 = np.random.choice(population[:10]) # Pick from top 10
            p2 = np.random.choice(population[:10])
            
            parent = p1 if p1['fitness'] > p2['fitness'] else p2
            
            # Create child (Clone + Mutate)
            child_nn = parent['nn'].copy()
            child_nn.mutate(config.MUTATION_RATE)
            
            new_population.append({'nn': child_nn, 'fitness': 0})
            
        population = new_population[:config.POPULATION_SIZE]
        generation += 1

if __name__ == "__main__":
    main()

