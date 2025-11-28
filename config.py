import math

# Screen settings
SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 600
FPS = 60

# Physics settings
GRAVITY = (0.0, -900.0)  # Pymunk uses y-up? No, usually y-down. Let's check Pymunk coordinate system. 
# Pymunk is usually y-up logic, but pygame is y-down. We'll handle conversion.
# Let's use standard gravity 9.8 scaled.

# Bike settings
CHASSIS_MASS = 20
WHEEL_MASS = 5
WHEEL_RADIUS = 25
CHASSIS_WIDTH = 80
CHASSIS_HEIGHT = 20
MOTOR_SPEED = -60  # Negative for forward in pymunk if wheels rotate counter-clockwise?

# Raycasting settings
RAY_COUNT = 5
RAY_LENGTH = 300
RAY_SPREAD = math.pi / 2  # 90 degrees spread

# GA settings
POPULATION_SIZE = 20
MUTATION_RATE = 0.1
ELITISM_COUNT = 2

