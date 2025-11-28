# AI Learning to Ride Bike with Raycasting

This project implements a genetic algorithm where a neural network learns to ride a bike over procedural terrain.

## Features
- **Physics**: Uses Pymunk (Chipmunk2D) for realistic rigid body physics.
- **Neural Network**: A Feedforward NN controls the bike.
- **Genetic Algorithm**: Evolves the population over generations.
- **Vision**: The bike uses 5 raycasts (Lidar) to see the terrain ahead, allowing it to anticipate hills and jumps.

## Installation

1. Install Python 3.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the simulation:
```bash
python main.py
```

## How "Vision" Works
The bike casts 5 rays in front of it. The normalized distance to the terrain is fed into the neural network along with the bike's internal state (angle, velocity, etc.). This allows the AI to learn behaviors like:
- Leaning forward before a hill climb.
- Stabilizing before a landing.

