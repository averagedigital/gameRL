import pymunk
import pygame
import math
import numpy as np
import config

class Bike:
    def __init__(self, space, start_pos):
        self.space = space
        self.alive = True
        self.distance_traveled = 0
        self.time_alive = 0
        
        # Create Bike parts
        x, y = start_pos
        
        # Chassis
        self.chassis_body = pymunk.Body(config.CHASSIS_MASS, pymunk.moment_for_box(config.CHASSIS_MASS, (config.CHASSIS_WIDTH, config.CHASSIS_HEIGHT)))
        self.chassis_body.position = (x, y)
        self.chassis_shape = pymunk.Poly.create_box(self.chassis_body, (config.CHASSIS_WIDTH, config.CHASSIS_HEIGHT))
        self.chassis_shape.friction = 0.5
        self.chassis_shape.filter = pymunk.ShapeFilter(group=1)
        self.chassis_shape.color = (200, 50, 50, 255)
        
        # Wheels
        wheel_offset_x = config.CHASSIS_WIDTH / 2
        wheel_offset_y = -config.CHASSIS_HEIGHT / 2
        
        self.rear_wheel_body = pymunk.Body(config.WHEEL_MASS, pymunk.moment_for_circle(config.WHEEL_MASS, 0, config.WHEEL_RADIUS))
        self.rear_wheel_body.position = (x - wheel_offset_x, y + wheel_offset_y)
        self.rear_wheel_shape = pymunk.Circle(self.rear_wheel_body, config.WHEEL_RADIUS)
        self.rear_wheel_shape.friction = 0.9
        self.rear_wheel_shape.filter = pymunk.ShapeFilter(group=1)
        self.rear_wheel_shape.color = (50, 200, 50, 255)

        self.front_wheel_body = pymunk.Body(config.WHEEL_MASS, pymunk.moment_for_circle(config.WHEEL_MASS, 0, config.WHEEL_RADIUS))
        self.front_wheel_body.position = (x + wheel_offset_x, y + wheel_offset_y)
        self.front_wheel_shape = pymunk.Circle(self.front_wheel_body, config.WHEEL_RADIUS)
        self.front_wheel_shape.friction = 0.9
        self.front_wheel_shape.filter = pymunk.ShapeFilter(group=1)
        self.front_wheel_shape.color = (50, 200, 50, 255)
        
        # Joints (Suspension/Connection)
        # Using PinJoints and DampedSprings for suspension effect
        # Rear
        self.rear_joint = pymunk.PinJoint(self.chassis_body, self.rear_wheel_body, (-wheel_offset_x, wheel_offset_y), (0,0))
        self.rear_spring = pymunk.DampedSpring(self.chassis_body, self.rear_wheel_body, (-wheel_offset_x, wheel_offset_y), (0,0), rest_length=0, stiffness=500, damping=20)
        
        # Front
        self.front_joint = pymunk.PinJoint(self.chassis_body, self.front_wheel_body, (wheel_offset_x, wheel_offset_y), (0,0))
        self.front_spring = pymunk.DampedSpring(self.chassis_body, self.front_wheel_body, (wheel_offset_x, wheel_offset_y), (0,0), rest_length=0, stiffness=500, damping=20)
        
        # Motor (SimpleMotor on rear wheel)
        self.motor = pymunk.SimpleMotor(self.chassis_body, self.rear_wheel_body, 0)
        self.motor.max_force = 1000000
        
        # Add to space
        space.add(self.chassis_body, self.chassis_shape)
        space.add(self.rear_wheel_body, self.rear_wheel_shape)
        space.add(self.front_wheel_body, self.front_wheel_shape)
        space.add(self.rear_joint, self.front_joint, self.rear_spring, self.front_spring, self.motor)
        
    def update(self, action):
        if not self.alive:
            return
            
        # Action: [Gas/Brake, Lean]
        # Gas: -1 to 1 (mapped to motor rate)
        # Lean: -1 to 1 (mapped to torque on chassis)
        
        gas = action[0]
        lean = action[1]
        
        self.motor.rate = gas * config.MOTOR_SPEED
        self.chassis_body.apply_force_at_local_point((0, lean * 10000), (config.CHASSIS_WIDTH/2, 0)) # Fake lean torque
        self.chassis_body.torque = lean * 200000 # Direct torque is better
        
        # Update fitness metrics
        self.distance_traveled = self.chassis_body.position.x
        self.time_alive += 1
        
        # Death condition: Chassis touches ground or head hits
        # We handle collisions in the main loop or here via checking angle
        if abs(self.chassis_body.angle) > math.pi / 2 + 0.5: # Upside downish
            self.alive = False

    def raycast(self):
        # Cast rays from chassis center forward
        start_point = self.chassis_body.position
        angle_base = self.chassis_body.angle
        
        distances = []
        
        # Create rays
        # We want rays spreading in front of the bike
        # -45 to +45 degrees relative to bike forward
        
        for i in range(config.RAY_COUNT):
            # Calculate ray angle
            # Map i from 0..RAY_COUNT-1 to -SPREAD/2 .. +SPREAD/2
            relative_angle = -config.RAY_SPREAD/2 + (config.RAY_SPREAD / (config.RAY_COUNT - 1)) * i
            ray_angle = angle_base + relative_angle
            
            end_point = (
                start_point.x + math.cos(ray_angle) * config.RAY_LENGTH,
                start_point.y + math.sin(ray_angle) * config.RAY_LENGTH
            )
            
            # Query the space
            # filter=pymunk.ShapeFilter(mask=pymunk.ShapeFilter.ALL_MASKS ^ 1) # Ignore bike group 1
            # Actually, terrain should be in a specific group or default.
            # We want to ignore the bike itself.
            
            info = self.space.segment_query_first(start_point, end_point, 1, pymunk.ShapeFilter(mask=0b10)) # Mask for terrain only
            
            if info:
                dist = np.linalg.norm(np.array(info.point) - np.array(start_point))
                distances.append(dist / config.RAY_LENGTH) # Normalize 0..1
                
                # Visual debugging data (not used for NN)
                self.last_rays = getattr(self, 'last_rays', [])
                if i == 0: self.last_rays = []
                self.last_rays.append((start_point, info.point))
            else:
                distances.append(1.0) # Max distance
                
                self.last_rays = getattr(self, 'last_rays', [])
                if i == 0: self.last_rays = []
                self.last_rays.append((start_point, end_point))

        return distances

    def get_state(self):
        # 5 Basic Sensors + Raycasts
        # 1. Angle (normalized -pi to pi -> -1 to 1)
        angle = self.chassis_body.angle / math.pi
        
        # 2. Angular Velocity
        ang_vel = self.chassis_body.angular_velocity / 10.0
        
        # 3. Vertical Velocity
        vel_y = self.chassis_body.velocity.y / 100.0
        
        # 4. Horizontal Velocity
        vel_x = self.chassis_body.velocity.x / 100.0
        
        # 5. Height (relative to... something? or absolute normalized)
        # For infinite runners, absolute y is okay if terrain varies.
        height = self.chassis_body.position.y / 500.0 
        
        basic_sensors = [angle, ang_vel, vel_y, vel_x, height]
        
        # Vision Sensors
        vision = self.raycast()
        
        return np.array(basic_sensors + vision)

    def destroy(self):
        self.space.remove(self.chassis_body, self.chassis_shape)
        self.space.remove(self.rear_wheel_body, self.rear_wheel_shape)
        self.space.remove(self.front_wheel_body, self.front_wheel_shape)
        self.space.remove(self.rear_joint, self.front_joint, self.rear_spring, self.front_spring, self.motor)

class Terrain:
    def __init__(self, space):
        self.space = space
        self.body = space.static_body
        self.shapes = []
        self.generate()
        
    def generate(self):
        # Simple procedural terrain
        points = []
        segment_length = 50
        num_segments = 200
        
        x = 0
        y = 100
        points.append((x, 0)) # Bottom left anchor
        points.append((x, y)) # Start
        
        for i in range(num_segments):
            x += segment_length
            # Random height change
            y += np.random.randint(-30, 30)
            if y < 50: y = 50 # Floor
            if y > 500: y = 500
            
            # Add hills occasionally
            if i > 10 and i % 20 < 10:
                 y += 30
            
            points.append((x, y))
            
        points.append((x, 0)) # Bottom right anchor
        
        # Create shapes
        for i in range(len(points) - 1):
             # Pymunk segments or poly? Poly is safer for catching ghost collisions
             # But a single massive poly is bad.
             # Let's use Segment shapes for the surface.
             p1 = points[i]
             p2 = points[i+1]
             # To make it solid, let's use thick segments or polygons.
             # Segment is fine for wheels usually.
             
             # Actually, let's just make lines.
             if i > 0 and i < len(points)-1: # Skip the closing loop for lines
                shape = pymunk.Segment(self.body, points[i], points[i+1], 5)
                shape.friction = 1.0
                shape.filter = pymunk.ShapeFilter(categories=0b10) # Category 2
                self.space.add(shape)
                self.shapes.append(shape)
        
        self.points = points


