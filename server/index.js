const express = require('express');
const http = require('http');
const { Server } = require('socket.io');
const Matter = require('matter-js');

const app = express();
const server = http.createServer(app);
const io = new Server(server, {
    cors: { origin: "*" }
});

// --- PHYSICS ENGINE (Server Side) ---
const Engine = Matter.Engine,
      World = Matter.World,
      Bodies = Matter.Bodies,
      Body = Matter.Body,
      Composite = Matter.Composite,
      Constraint = Matter.Constraint,
      Query = Matter.Query,
      Vector = Matter.Vector;

const engine = Engine.create();
const world = engine.world;
engine.gravity.y = 1; // GRAVITY_Y

// --- CONSTANTS ---
const WORLD_WIDTH = 30000;
const WORLD_HEIGHT = 2000;
const CHASSIS_WIDTH = 80;
const CHASSIS_HEIGHT = 20;
const WHEEL_RADIUS = 25;
const MOTOR_SPEED = 0.5;
const POPULATION_SIZE = 12;
const MUTATION_RATE = 0.1;
const RAY_COUNT = 5;
const RAY_LENGTH = 300;
const RAY_SPREAD = Math.PI / 2;

// --- STATE ---
let population = [];
let terrainBody;
let terrainVertices = [];
let generation = 0;
let frameCount = 0;
const MAX_FRAMES = 60 * 120; // 2 mins

// --- NEURAL NETWORK CLASS ---
class NeuralNetwork {
    constructor(inputSize, hiddenSize, outputSize) {
        this.inputSize = inputSize;
        this.hiddenSize = hiddenSize;
        this.outputSize = outputSize;
        this.w1 = this.randomMatrix(inputSize, hiddenSize);
        this.b1 = this.randomMatrix(1, hiddenSize);
        this.w2 = this.randomMatrix(hiddenSize, outputSize);
        this.b2 = this.randomMatrix(1, outputSize);
    }
    randomMatrix(rows, cols) {
        let m = [];
        for(let i=0; i<rows; i++) {
            m[i] = [];
            for(let j=0; j<cols; j++) m[i][j] = Math.random() * 2 - 1;
        }
        return m;
    }
    matmul(a, b) {
        let aRows = a.length, aCols = a[0].length, bCols = b[0].length, m = new Array(aRows);
        for (let r = 0; r < aRows; ++r) {
            m[r] = new Array(bCols);
            for (let c = 0; c < bCols; ++c) {
                m[r][c] = 0;
                for (let i = 0; i < aCols; ++i) m[r][c] += a[r][i] * b[i][c];
            }
        }
        return m;
    }
    addBias(m, b) {
        let res = [];
        for(let i=0; i<m.length; i++) {
            res[i] = [];
            for(let j=0; j<m[0].length; j++) res[i][j] = m[i][j] + b[0][j];
        }
        return res;
    }
    tanh(m) { return m.map(row => row.map(val => Math.tanh(val))); }
    predict(inputs) {
        let h1 = this.matmul([inputs], this.w1);
        h1 = this.tanh(this.addBias(h1, this.b1));
        let out = this.tanh(this.addBias(this.matmul(h1, this.w2), this.b2));
        return out[0];
    }
    copy() {
        const nn = new NeuralNetwork(this.inputSize, this.hiddenSize, this.outputSize);
        nn.w1 = JSON.parse(JSON.stringify(this.w1));
        nn.b1 = JSON.parse(JSON.stringify(this.b1));
        nn.w2 = JSON.parse(JSON.stringify(this.w2));
        nn.b2 = JSON.parse(JSON.stringify(this.b2));
        return nn;
    }
    mutate(rate) {
        this.mutateMatrix(this.w1, rate);
        this.mutateMatrix(this.b1, rate);
        this.mutateMatrix(this.w2, rate);
        this.mutateMatrix(this.b2, rate);
    }
    mutateMatrix(m, rate) {
        for(let i=0; i<m.length; i++) {
            for(let j=0; j<m[0].length; j++) {
                if(Math.random() < rate) m[i][j] += (Math.random() * 2 - 1) * 0.5;
            }
        }
    }
}

// --- BIKE CLASS ---
class Bike {
    constructor(x, y, brain) {
        this.brain = brain || new NeuralNetwork(5 + RAY_COUNT, 8, 2);
        this.fitness = 0;
        this.alive = true;
        this.distance = 0;
        this.idleTime = 0;
        
        // GHOST MODE FILTER
        const filter = {
            group: Body.nextGroup(true),
            category: 0x0002,
            mask: 0x0001 // Terrain only
        };

        this.chassis = Bodies.rectangle(x, y, CHASSIS_WIDTH, CHASSIS_HEIGHT, { collisionFilter: filter, density: 0.04 });
        this.wheelBack = Bodies.circle(x - 30, y + 20, WHEEL_RADIUS, { collisionFilter: filter, friction: 0.9, density: 0.05, restitution: 0.2 });
        this.wheelFront = Bodies.circle(x + 30, y + 20, WHEEL_RADIUS, { collisionFilter: filter, friction: 0.9, density: 0.05, restitution: 0.2 });
        
        this.axleBack = Constraint.create({ bodyA: this.chassis, bodyB: this.wheelBack, pointA: {x:-30,y:15}, stiffness: 0.2, damping: 0.05, length: 0 });
        this.axleFront = Constraint.create({ bodyA: this.chassis, bodyB: this.wheelFront, pointA: {x:30,y:15}, stiffness: 0.2, damping: 0.05, length: 0 });

        this.composite = Composite.create();
        Composite.add(this.composite, [this.chassis, this.wheelBack, this.wheelFront, this.axleBack, this.axleFront]);
    }

    addToWorld(w) { Composite.add(w, this.composite); }
    removeFromWorld(w) { Composite.remove(w, this.composite); }

    getLineIntersection(p0, p1, p2, p3) {
        const s1_x = p1.x - p0.x, s1_y = p1.y - p0.y;
        const s2_x = p3.x - p2.x, s2_y = p3.y - p2.y;
        const s = (-s1_y * (p0.x - p2.x) + s1_x * (p0.y - p2.y)) / (-s2_x * s1_y + s1_x * s2_y);
        const t = ( s2_x * (p0.y - p2.y) - s2_y * (p0.x - p2.x)) / (-s2_x * s1_y + s1_x * s2_y);
        if (s >= 0 && s <= 1 && t >= 0 && t <= 1) return { x: p0.x + (t * s1_x), y: p0.y + (t * s1_y) };
        return null;
    }

    castRays() {
        let start = this.chassis.position;
        let angleBase = this.chassis.angle;
        let rays = [];
        this.lastRays = []; // Store for serialization

        for(let i=0; i<RAY_COUNT; i++) {
            let relAngle = -RAY_SPREAD/2 + (RAY_SPREAD / (RAY_COUNT-1))*i;
            let angle = angleBase + relAngle;
            let dir = { x: Math.cos(angle), y: Math.sin(angle) };
            
            // Raycast against terrain segments
            let minDist = RAY_LENGTH;
            let hit = false;
            let endP = { x: start.x + dir.x * RAY_LENGTH, y: start.y + dir.y * RAY_LENGTH };

            // Optimization: check close segments
            let startIdx = Math.floor(start.x / 50) - 2;
            if(startIdx < 0) startIdx = 0;
            let endIdx = startIdx + 15;
            if(endIdx > terrainVertices.length - 2) endIdx = terrainVertices.length - 2;

            for(let j=startIdx; j<=endIdx; j++) {
                let p1 = terrainVertices[j], p2 = terrainVertices[j+1];
                let pt = this.getLineIntersection(start, endP, p1, p2);
                if(pt) {
                    let d = Math.sqrt((pt.x-start.x)**2 + (pt.y-start.y)**2);
                    if(d < minDist) { minDist = d; endP = pt; hit = true; }
                }
            }
            rays.push(minDist / RAY_LENGTH);
            this.lastRays.push({ start, end: endP, hit });
        }
        return rays;
    }

    update() {
        if(!this.alive) return;
        
        let rays = this.castRays();
        let angle = this.chassis.angle / Math.PI;
        let angVel = Math.max(-1, Math.min(1, this.chassis.angularVelocity * 5));
        let vX = Math.max(-1, Math.min(1, this.chassis.velocity.x / 20));
        let vY = Math.max(-1, Math.min(1, this.chassis.velocity.y / 10));
        let height = (600 - this.chassis.position.y) / 600;

        let inputs = [angle, angVel, vX, vY, height, ...rays];
        let output = this.brain.predict(inputs);
        let gas = output[0];
        let lean = output[1];

        // Physics
        if(gas !== 0) {
            let speed = MOTOR_SPEED * 2; // AI gets boost
            Body.setAngularVelocity(this.wheelBack, this.wheelBack.angularVelocity + gas * speed);
        } else {
            Body.setAngularVelocity(this.wheelBack, this.wheelBack.angularVelocity * 0.9);
        }

        let leanPower = 0.08;
        if(lean !== 0) {
            Body.applyForce(this.chassis, this.chassis.position, {x: lean * 0.002, y: 0});
        }
        Body.setAngularVelocity(this.chassis, this.chassis.angularVelocity + lean * leanPower);

        // Update Fitness
        this.distance = this.chassis.position.x;
        this.fitness = this.distance;

        // Death Conditions
        if(Math.abs(this.chassis.angle) > Math.PI / 2) this.alive = false;
        
        let velMag = Math.sqrt(this.chassis.velocity.x**2 + this.chassis.velocity.y**2);
        if(velMag < 0.5) {
            this.idleTime++;
            if(this.idleTime > 180) this.alive = false;
        } else {
            this.idleTime = 0;
        }
    }
}

// --- GAME LOGIC ---
function generateTerrain() {
    if(terrainBody) Composite.remove(world, terrainBody);
    terrainVertices = [];
    let x = 0;
    let y = 500;
    
    // 1000px flat start
    for(let i=0; i<300; i++) {
        terrainVertices.push({x, y});
        x += 50;
        if(i > 20) {
            y += (Math.random() - 0.5) * 60;
            if(y > 800) y = 800;
            if(y < 200) y = 200;
            if(i > 30 && i % 20 < 10) y -= 40;
        }
    }
    
    // Build physics bodies
    terrainBody = Composite.create();
    for(let i=0; i<terrainVertices.length-1; i++) {
        let p1 = terrainVertices[i], p2 = terrainVertices[i+1];
        let cx = (p1.x+p2.x)/2, cy = (p1.y+p2.y)/2;
        let len = Math.sqrt((p2.x-p1.x)**2 + (p2.y-p1.y)**2);
        let angle = Math.atan2(p2.y-p1.y, p2.x-p1.x);
        let seg = Bodies.rectangle(cx, cy, len, 10, { isStatic: true, friction: 1, angle: angle });
        Composite.add(terrainBody, seg);
    }
    Composite.add(world, terrainBody);
}

function createPopulation(brains = null) {
    population.forEach(b => b.removeFromWorld(world));
    population = [];
    
    for(let i=0; i<POPULATION_SIZE; i++) {
        let brain = brains ? brains[i] : null;
        let b = new Bike(300, 300, brain);
        b.addToWorld(world);
        population.push(b);
    }
}

function nextGeneration() {
    generation++;
    frameCount = 0;
    population.sort((a,b) => b.fitness - a.fitness);
    
    let newBrains = [];
    // Elitism
    for(let i=0; i<2; i++) if(population[i]) newBrains.push(population[i].brain.copy());
    
    // Selection
    while(newBrains.length < POPULATION_SIZE) {
        let p1 = population[Math.floor(Math.random()*population.length)];
        let p2 = population[Math.floor(Math.random()*population.length)];
        if(!p1) p1 = population[0]; if(!p2) p2 = population[0];
        let parent = (p1.fitness > p2.fitness) ? p1 : p2;
        let child = parent.brain.copy();
        child.mutate(MUTATION_RATE);
        newBrains.push(child);
    }
    
    createPopulation(newBrains);
    generateTerrain();
    
    io.emit('new_generation', { generation, terrain: terrainVertices });
}

// --- SERVER LOOP ---
generateTerrain();
createPopulation();

setInterval(() => {
    Engine.update(engine, 1000/60);
    
    let aliveCount = 0;
    let packet = []; // Data to send to clients
    
    population.forEach(b => {
        b.update();
        if(b.alive) aliveCount++;
        
        // Optimize packet
        packet.push({
            x: Math.round(b.chassis.position.x),
            y: Math.round(b.chassis.position.y),
            a: parseFloat(b.chassis.angle.toFixed(2)),
            // Send Wheel Positions for proper rendering
            wb_x: Math.round(b.wheelBack.position.x),
            wb_y: Math.round(b.wheelBack.position.y),
            wb_a: parseFloat(b.wheelBack.angle.toFixed(2)),
            wf_x: Math.round(b.wheelFront.position.x),
            wf_y: Math.round(b.wheelFront.position.y),
            wf_a: parseFloat(b.wheelFront.angle.toFixed(2)),
            alive: b.alive,
            rays: b.lastRays
        });
    });
    
    frameCount++;
    if(frameCount > 60) { // Safety buffer
        if(aliveCount === 0 || frameCount > MAX_FRAMES) {
            nextGeneration();
            return;
        }
    }
    
    // Broadcast state
    io.emit('state', { bikes: packet, frame: frameCount });
    
}, 1000/60);

// --- API ---
io.on('connection', (socket) => {
    console.log('Client connected');
    // Send initial state
    socket.emit('init', { 
        terrain: terrainVertices, 
        generation: generation 
    });
    
    // Handle Human Inputs (Optional Future Feature)
});

const PORT = process.env.PORT || 3000;
server.listen(PORT, () => {
    console.log(`Game Server running on port ${PORT}`);
});

