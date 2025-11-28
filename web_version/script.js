// Constants
const WORLD_WIDTH = 30000;
const WORLD_HEIGHT = 2000;
const VIEWPORT_WIDTH = window.innerWidth;
const VIEWPORT_HEIGHT = window.innerHeight;

// Physics Config
const GRAVITY_Y = 1;
const CHASSIS_WIDTH = 80;
const CHASSIS_HEIGHT = 20;
const WHEEL_RADIUS = 25;
const MOTOR_SPEED = 0.3;

// GA Config
const POPULATION_SIZE = 20;
const MUTATION_RATE = 0.1;
const RAY_COUNT = 5;
const RAY_LENGTH = 300;
const RAY_SPREAD = Math.PI / 2;

// Matter.js aliases
const Engine = Matter.Engine,
      Render = Matter.Render,
      Runner = Matter.Runner,
      Composite = Matter.Composite,
      Composites = Matter.Composites,
      Common = Matter.Common,
      Events = Matter.Events,
      MouseConstraint = Matter.MouseConstraint,
      Mouse = Matter.Mouse,
      Body = Matter.Body,
      Bodies = Matter.Bodies,
      Vector = Matter.Vector,
      Constraint = Matter.Constraint,
      Query = Matter.Query;

// Neural Network Class
class NeuralNetwork {
    constructor(inputSize, hiddenSize, outputSize) {
        this.inputSize = inputSize;
        this.hiddenSize = hiddenSize;
        this.outputSize = outputSize;
        
        // Weights and Biases
        this.w1 = this.randomMatrix(inputSize, hiddenSize);
        this.b1 = this.randomMatrix(1, hiddenSize);
        this.w2 = this.randomMatrix(hiddenSize, outputSize);
        this.b2 = this.randomMatrix(1, outputSize);
    }

    randomMatrix(rows, cols) {
        let matrix = [];
        for(let i=0; i<rows; i++) {
            matrix[i] = [];
            for(let j=0; j<cols; j++) {
                matrix[i][j] = Math.random() * 2 - 1;
            }
        }
        return matrix;
    }

    predict(inputs) {
        // Simple matrix multiplication + activation
        // Inputs is 1xN array
        
        // Hidden Layer
        let h1 = this.matmul([inputs], this.w1); // Result 1xHidden
        h1 = this.addBias(h1, this.b1);
        h1 = this.tanh(h1);
        
        // Output Layer
        let out = this.matmul(h1, this.w2); // Result 1xOutput
        out = this.addBias(out, this.b2);
        out = this.tanh(out);
        
        return out[0];
    }
    
    matmul(a, b) {
        let aRows = a.length, aCols = a[0].length,
            bRows = b.length, bCols = b[0].length,
            m = new Array(aRows);
        for (let r = 0; r < aRows; ++r) {
            m[r] = new Array(bCols);
            for (let c = 0; c < bCols; ++c) {
                m[r][c] = 0;
                for (let i = 0; i < aCols; ++i) {
                    m[r][c] += a[r][i] * b[i][c];
                }
            }
        }
        return m;
    }
    
    addBias(matrix, bias) {
        let res = [];
        for(let i=0; i<matrix.length; i++) {
            res[i] = [];
            for(let j=0; j<matrix[0].length; j++) {
                res[i][j] = matrix[i][j] + bias[0][j];
            }
        }
        return res;
    }
    
    tanh(matrix) {
        return matrix.map(row => row.map(val => Math.tanh(val)));
    }

    copy() {
        let nn = new NeuralNetwork(this.inputSize, this.hiddenSize, this.outputSize);
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
    
    mutateMatrix(matrix, rate) {
        for(let i=0; i<matrix.length; i++) {
            for(let j=0; j<matrix[0].length; j++) {
                if(Math.random() < rate) {
                    matrix[i][j] += (Math.random() * 2 - 1) * 0.5;
                }
            }
        }
    }
}

// Bike Class
class Bike {
    constructor(x, y, brain, isHuman = false) {
        this.isHuman = isHuman;
        this.brain = brain || new NeuralNetwork(5 + RAY_COUNT, 8, 2);
        this.fitness = 0;
        this.alive = true;
        this.distance = 0;
        
        const group = Body.nextGroup(true);
        
        // Chassis
        this.chassis = Bodies.rectangle(x, y, CHASSIS_WIDTH, CHASSIS_HEIGHT, { 
            collisionFilter: { group: group },
            density: 0.04,
            friction: 0.5,
            label: 'chassis'
        });
        
        // Wheels
        this.wheelBack = Bodies.circle(x - 30, y + 20, WHEEL_RADIUS, { 
            collisionFilter: { group: group },
            friction: 0.9,
            density: 0.05,
            restitution: 0.2
        });
        
        this.wheelFront = Bodies.circle(x + 30, y + 20, WHEEL_RADIUS, { 
            collisionFilter: { group: group },
            friction: 0.9,
            density: 0.05,
            restitution: 0.2
        });
        
        // Constraints (Suspension)
        this.axleBack = Constraint.create({
            bodyA: this.chassis,
            bodyB: this.wheelBack,
            pointA: { x: -30, y: 15 },
            stiffness: 0.2,
            damping: 0.05,
            length: 0
        });
        
        this.axleFront = Constraint.create({
            bodyA: this.chassis,
            bodyB: this.wheelFront,
            pointA: { x: 30, y: 15 },
            stiffness: 0.2,
            damping: 0.05,
            length: 0
        });

        this.composite = Composite.create();
        Composite.add(this.composite, [this.chassis, this.wheelBack, this.wheelFront, this.axleBack, this.axleFront]);
    }
    
    addToWorld(world) {
        Composite.add(world, this.composite);
    }
    
    removeFromWorld(world) {
        Composite.remove(world, this.composite);
    }
    
    update(terrainBodies) {
        if(!this.alive) return;
        
        let gas = 0;
        let lean = 0;

        // Rays always needed for visualization
        const rays = this.castRays(terrainBodies);

        if (this.isHuman) {
            // Human Control
            // Swapped Controls as requested:
            // Arrow Down / S = Gas (Forward)
            // Arrow Up / W = Brake/Back
            // Arrow Right / D = Lean Forward
            // Arrow Left / A = Lean Back
            
            if (keys['ArrowDown'] || keys['KeyS']) gas = 1;
            if (keys['ArrowUp'] || keys['KeyW']) gas = -1;
            if (keys['ArrowRight'] || keys['KeyD']) lean = 1;   // Lean forward
            if (keys['ArrowLeft'] || keys['KeyA']) lean = -1; // Lean back
        } else {
            // AI Control
            // Inputs
            const angle = this.chassis.angle / Math.PI; // Normalized
            const angVel = this.chassis.angularVelocity;
            const vX = this.chassis.velocity.x;
            const vY = this.chassis.velocity.y;
            const height = (600 - this.chassis.position.y) / 600;
            
            const inputs = [angle, angVel, vX, vY, height, ...rays];
            const output = this.brain.predict(inputs);
            
            gas = output[0]; // -1 to 1
            lean = output[1]; // -1 to 1
        }
        
        // Apply Motor
        // We apply torque directly to wheel or AngularVelocity
        if (gas !== 0) {
            // Forward (negative torque/velocity in matter.js usually? depends on coordinate)
            // Matter.js clockwise is positive. So forward is clockwise?
            // Let's try setting angular velocity target
           Body.setAngularVelocity(this.wheelBack, this.wheelBack.angularVelocity + gas * MOTOR_SPEED);
        }
        
        // Apply Lean Torque
        Body.setAngularVelocity(this.chassis, this.chassis.angularVelocity + lean * 0.05);
        
        // Update Fitness
        this.distance = this.chassis.position.x;
        this.fitness = this.distance;
        
        // Death check
        if (Math.abs(this.chassis.angle) > Math.PI / 1.5) {
             this.alive = false;
        }
    }
    
    castRays(terrainBodies) {
        let distances = [];
        const start = this.chassis.position;
        const baseAngle = this.chassis.angle;
        
        this.lastRays = []; // For rendering
        
        for(let i=0; i<RAY_COUNT; i++) {
            const relAngle = -RAY_SPREAD/2 + (RAY_SPREAD / (RAY_COUNT - 1)) * i;
            const angle = baseAngle + relAngle;
            
            const end = {
                x: start.x + Math.cos(angle) * RAY_LENGTH,
                y: start.y + Math.sin(angle) * RAY_LENGTH
            };
            
            const collisions = Query.ray(terrainBodies, start, end);
            
            if (collisions.length > 0) {
                // Find closest
                let minDist = RAY_LENGTH;
                let closest = null;
                
                collisions.forEach(col => {
                    const d = Vector.magnitude(Vector.sub(col.bodyA.position, start)); // Approx
                    // A better way is to use the collision point, but Matter.Query.ray returns bodies, not points directly easily without ray lib
                    // Actually Query.ray result has 'body' and logic.
                    // Matter.Query.ray checks AABBs then SAT. It doesn't give exact point easily.
                    // For simplified raycast, let's just use the body center distance or implement a simple segment intersection if we want precision.
                    // For now, let's trust the collision detection is boolean-ish and maybe just return 0.5 if hit?
                    // No, we need distance.
                    
                    // Let's use Raycasting from a library snippet or approximation.
                    // Approximation: Just check points along the line? Too slow.
                    // Let's assume if we hit something, it's "close". 
                    // To do it properly:
                    // We will implement a simple line-segment intersection with terrain segments.
                });
                
                // Let's simplify: Terrain is a set of vertices.
                // We can check intersection with terrain lines.
                // Assuming terrainBodies[0] is our ground.
                let hitDist = RAY_LENGTH;
                let hitPoint = end;
                
                // Optimized: The terrain is generated as vertices.
                // Let's iterate relevant segments.
                // This is heavy for JS.
                // Let's stick to simple raycast: 
                // Return 1 if clear, 0 if blocked? No, we need gradient.
                
                // Alternate: use Matter.Query.ray but improved.
                // Actually, let's just visualize "hit" or "miss" for now to save FPS, 
                // or implement proper math.
                
                // Simple implementation:
                distances.push(0.5); // Placeholder for "Obstacle Detected"
                this.lastRays.push({start, end, hit: true});
            } else {
                distances.push(1);
                this.lastRays.push({start, end, hit: false});
            }
        }
        
        // To make it REAL: we need actual raycasting against terrain vertices.
        // We can access terrain vertices from global scope or passed in.
        return this.realRaycast(start, baseAngle);
    }

    realRaycast(start, baseAngle) {
        // We will access the global 'terrainVertices'
        if (!window.terrainVertices) return new Array(RAY_COUNT).fill(1);
        
        let distances = [];
        this.lastRays = [];
        
        for(let i=0; i<RAY_COUNT; i++) {
            const relAngle = -RAY_SPREAD/2 + (RAY_SPREAD / (RAY_COUNT - 1)) * i;
            const angle = baseAngle + relAngle;
            
            const dir = { x: Math.cos(angle), y: Math.sin(angle) };
            
            let minDist = RAY_LENGTH;
            let bestPoint = { x: start.x + dir.x * RAY_LENGTH, y: start.y + dir.y * RAY_LENGTH };
            
            // Check intersection with terrain segments
            // Optimization: Only check segments within X range
            // Terrain is sorted by X.
            
            // Find start index
            let startIdx = Math.floor(start.x / 50) - 2;
            if(startIdx < 0) startIdx = 0;
            let endIdx = startIdx + 10; // Check next 10 segments (500px)
            if(endIdx > window.terrainVertices.length - 2) endIdx = window.terrainVertices.length - 2;
            
            for(let j=startIdx; j <= endIdx; j++) {
                const p1 = window.terrainVertices[j];
                const p2 = window.terrainVertices[j+1];
                
                const pt = this.getLineIntersection(start, {x: start.x + dir.x * RAY_LENGTH, y: start.y + dir.y * RAY_LENGTH}, p1, p2);
                if (pt) {
                    const d = Math.sqrt((pt.x - start.x)**2 + (pt.y - start.y)**2);
                    if (d < minDist) {
                        minDist = d;
                        bestPoint = pt;
                    }
                }
            }
            
            distances.push(minDist / RAY_LENGTH);
            this.lastRays.push({start, end: bestPoint, hit: minDist < RAY_LENGTH});
        }
        return distances;
    }
    
    getLineIntersection(p0, p1, p2, p3) {
        const s1_x = p1.x - p0.x;
        const s1_y = p1.y - p0.y;
        const s2_x = p3.x - p2.x;
        const s2_y = p3.y - p2.y;
        
        const s = (-s1_y * (p0.x - p2.x) + s1_x * (p0.y - p2.y)) / (-s2_x * s1_y + s1_x * s2_y);
        const t = ( s2_x * (p0.y - p2.y) - s2_y * (p0.x - p2.x)) / (-s2_x * s1_y + s1_x * s2_y);
        
        if (s >= 0 && s <= 1 && t >= 0 && t <= 1) {
            return { x: p0.x + (t * s1_x), y: p0.y + (t * s1_y) };
        }
        return null;
    }
}

// Main Game Logic
let engine, world;
let population = [];
let terrainBody;
let generation = 0;
let simulationSpeed = 1;
let isHumanMode = false;
let humanBike = null;
let keys = {};

function init() {
    // Engine setup
    engine = Engine.create();
    world = engine.world;
    engine.gravity.y = GRAVITY_Y;
    
    // Canvas setup
    const canvas = document.getElementById('world');
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
    const ctx = canvas.getContext('2d');
    
    // Input Handling
    window.addEventListener('keydown', e => {
        keys[e.code] = true;
        if(isHumanMode && e.code === 'KeyR') {
            respawnHuman();
        }
    });
    window.addEventListener('keyup', e => keys[e.code] = false);

    // Generate Terrain
    generateTerrain();
    
    // Create Population
    createPopulation();
    
    // Start loop
    requestAnimationFrame(loop);
    
    // UI Events
    document.getElementById('toggle-speed').onclick = () => {
        if(isHumanMode) return; // No speedup in human mode
        simulationSpeed = simulationSpeed === 1 ? 5 : 1;
        document.getElementById('toggle-speed').innerText = `Speed Up (x${simulationSpeed})`;
    };
    
    document.getElementById('reset').onclick = () => {
        resetGeneration();
        generation = 0;
        document.getElementById('gen').innerText = generation;
    };

    document.getElementById('human-mode').onclick = toggleHumanMode;
    document.getElementById('export-btn').onclick = exportBestBrain;
    document.getElementById('import-btn').onclick = importBrain;
}

function toggleHumanMode() {
    isHumanMode = !isHumanMode;
    const btn = document.getElementById('human-mode');
    
    if(isHumanMode) {
        btn.innerText = "🤖 Back to AI";
        btn.style.background = "#34495e";
        simulationSpeed = 1; // Force normal speed
        document.getElementById('toggle-speed').style.display = 'none';
        
        // Kill all AI, spawn 1 human bike
        population.forEach(b => b.removeFromWorld(world));
        population = [];
        respawnHuman();
        
        document.getElementById('stats').style.display = 'none';
    } else {
        btn.innerText = "🎮 Play Human";
        btn.style.background = "#e67e22";
        document.getElementById('toggle-speed').style.display = 'inline-block';
        document.getElementById('stats').style.display = 'block';
        resetGeneration();
    }
}

function respawnHuman() {
    window.respawnTimeout = null; // Clear timeout flag immediately
    
    // Clear ALL physics bodies related to bikes
    if (humanBike) {
        humanBike.removeFromWorld(world);
    }
    population.forEach(b => b.removeFromWorld(world));
    population = [];
    
    // Create fresh bike
    humanBike = new Bike(150, 200, null, true);
    humanBike.addToWorld(world);
    population = [humanBike];
    
    // Force physics update to ensure new body is registered
    // Sometimes Matter.js needs a kick if state was weird
    Engine.update(engine, 1000/60);
}

function exportBestBrain() {
    // Find best AI from previous or current gen
    // Since we constantly mutate, let's grab the current survivor with max distance
    // Or ideally, we should store the 'champion' separately.
    // For now, let's grab the leader.
    if(isHumanMode) {
        alert("Can't export human brain! Switch to AI mode.");
        return;
    }
    
    const best = population.reduce((prev, current) => (prev.distance > current.distance) ? prev : current);
    const data = JSON.stringify(best.brain);
    navigator.clipboard.writeText(data).then(() => {
        alert("Best Brain copied to clipboard! Paste this in Telegram.");
    });
}

function importBrain() {
    const data = prompt("Paste the Brain string here:");
    if(!data) return;
    
    try {
        const json = JSON.parse(data);
        // Create a new population based on this brain
        resetGeneration();
        population.forEach((b, i) => {
             // Keep slight mutation for variety, or exact copy for first one
             b.brain = new NeuralNetwork(5 + RAY_COUNT, 8, 2);
             // Manually inject weights
             b.brain.w1 = json.w1; 
             b.brain.b1 = json.b1; 
             b.brain.w2 = json.w2; 
             b.brain.b2 = json.b2;
             
             // Mutate everyone except the first one so we can evolve FROM this point
             if (i > 0) b.brain.mutate(MUTATION_RATE);
        });
        alert("Brain imported! Evolution will continue from this point.");
    } catch(e) {
        alert("Invalid Brain string!");
        console.error(e);
    }
}

function createPopulation() {
    for(let i=0; i<POPULATION_SIZE; i++) {
        const b = new Bike(150, 200);
        b.addToWorld(world);
        population.push(b);
    }
}

function generateTerrain() {
    if (terrainBody) Composite.remove(world, terrainBody);
    
    let x = 0;
    let y = 500;
    const points = [{x: 0, y: 1000}]; // Bottom left anchor
    points.push({x: 0, y: y}); // Start
    
    window.terrainVertices = []; // Store for raycasting
    
    for(let i=0; i<300; i++) {
        window.terrainVertices.push({x, y});
        x += 50;
        y += (Math.random() - 0.5) * 60;
        if (y > 800) y = 800;
        if (y < 200) y = 200;
        
        // Add hill
        if (i > 10 && i % 20 < 10) y -= 40;
        
        points.push({x, y});
    }
    
    window.terrainVertices.push({x, y}); // Last surface point
    
    points.push({x: x, y: 1000}); // Bottom right anchor
    
    // Create body
    terrainBody = Bodies.fromVertices(x/2, 1000, [points], {
        isStatic: true,
        friction: 1,
        label: 'terrain'
    }, true);
    
    // The center of fromVertices is tricky. 
    // Easier: Chain of rectangles or use the fromVertices result but position it correctly.
    // Matter.js fromVertices centers the body.
    // Let's align it.
    // Actually, Bodies.fromVertices decomposes concave shapes.
    // A simpler way for terrain in Matter.js is a series of static rectangles or a ground chain.
    
    // Let's redo terrain as simple trapezoids or thick lines to ensure smoothness.
    Composite.remove(world, terrainBody);
    terrainBody = Composite.create();
    
    for(let i=0; i<window.terrainVertices.length-1; i++) {
        const p1 = window.terrainVertices[i];
        const p2 = window.terrainVertices[i+1];
        
        const cx = (p1.x + p2.x) / 2;
        const cy = (p1.y + p2.y) / 2;
        const len = Math.sqrt((p2.x - p1.x)**2 + (p2.y - p1.y)**2);
        const angle = Math.atan2(p2.y - p1.y, p2.x - p1.x);
        
        const segment = Bodies.rectangle(cx, cy, len, 10, {
            isStatic: true,
            friction: 1,
            angle: angle,
            label: 'ground'
        });
        Composite.add(terrainBody, segment);
    }
    
    Composite.add(world, terrainBody);
}

function loop() {
    const canvas = document.getElementById('world');
    const ctx = canvas.getContext('2d');
    
    // Simulation Steps
    for(let s=0; s<simulationSpeed; s++) {
        Engine.update(engine, 1000 / 60);
        
        let aliveCount = 0;
        population.forEach(bike => {
            bike.update(terrainBody);
            if(bike.alive) aliveCount++;
        });
        
        if (aliveCount === 0) {
            nextGeneration();
            break;
        }
        
        // Timeout check (optional)
    }
    
    // Rendering
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Draw Wasted Overlay if human dead
    if (isHumanMode && humanBike && !humanBike.alive) {
         ctx.save();
         ctx.fillStyle = "rgba(0, 0, 0, 0.5)";
         ctx.fillRect(0, 0, canvas.width, canvas.height);
         ctx.fillStyle = "red";
         ctx.font = "bold 60px Arial";
         ctx.textAlign = "center";
         ctx.fillText("WASTED", canvas.width/2, canvas.height/2);
         ctx.fillStyle = "white";
         ctx.font = "20px Arial";
         ctx.fillText("Respawning...", canvas.width/2, canvas.height/2 + 40);
         ctx.restore();
         // Don't follow camera if wasted? Or keep following dead body?
         // Let's keep following dead body for drama, but we need to set camera transform BEFORE overlay?
         // No, overlay should be on top of everything. 
         // So we need to do standard rendering first, then overlay.
    }
    
    // Camera Follow logic moved before Overlay
    let bestBike = population.find(b => b.alive) || population[0];
    // In human mode, if dead, still follow the dead body
    if (isHumanMode && humanBike) bestBike = humanBike; 
    
    let maxDist = -Infinity;
    if (!isHumanMode) {
        population.forEach(b => {
            if(b.alive && b.distance > maxDist) {
                maxDist = b.distance;
                bestBike = b;
            }
        });
        if (!bestBike) bestBike = population[0];
    }
    
    ctx.save();
    // Center camera on best bike
    const camX = -bestBike.chassis.position.x + canvas.width / 3;
    ctx.translate(camX, 0);
    
    // Draw Terrain
    ctx.beginPath();
    ctx.strokeStyle = '#333';
    ctx.lineWidth = 5;
    if(window.terrainVertices) {
        ctx.moveTo(window.terrainVertices[0].x, window.terrainVertices[0].y);
        for(let i=1; i<window.terrainVertices.length; i++) {
            ctx.lineTo(window.terrainVertices[i].x, window.terrainVertices[i].y);
        }
    }
    ctx.stroke();
    
    // Draw Bikes
    population.forEach(bike => {
        // if(!bike.alive) return; // Draw dead ones too for physics debug
        
        // Chassis
        ctx.save();
        ctx.translate(bike.chassis.position.x, bike.chassis.position.y);
        ctx.rotate(bike.chassis.angle);
        ctx.fillStyle = bike.isHuman ? '#3498db' : '#e74c3c'; // Blue for human
        ctx.fillRect(-CHASSIS_WIDTH/2, -CHASSIS_HEIGHT/2, CHASSIS_WIDTH, CHASSIS_HEIGHT);
        ctx.restore();
        
        // Wheels
        [bike.wheelBack, bike.wheelFront].forEach(w => {
            ctx.beginPath();
            ctx.arc(w.position.x, w.position.y, WHEEL_RADIUS, 0, Math.PI * 2);
            ctx.fillStyle = '#34495e';
            ctx.fill();
            // Spoke
            ctx.beginPath();
            ctx.moveTo(w.position.x, w.position.y);
            ctx.lineTo(w.position.x + Math.cos(w.angle)*WHEEL_RADIUS, w.position.y + Math.sin(w.angle)*WHEEL_RADIUS);
            ctx.strokeStyle = '#fff';
            ctx.stroke();
        });
        
        // Rays (only draw if alive or debug)
        if(bike.alive && bike.lastRays) {
            bike.lastRays.forEach(ray => {
                ctx.beginPath();
                ctx.moveTo(ray.start.x, ray.start.y);
                ctx.lineTo(ray.end.x, ray.end.y);
                ctx.strokeStyle = ray.hit ? 'rgba(255, 0, 0, 0.5)' : 'rgba(0, 255, 0, 0.2)';
                ctx.lineWidth = 1;
                ctx.stroke();
            });
        }
    });
    
    ctx.restore();
    
    // UI Update
    document.getElementById('alive').innerText = population.filter(b => b.alive).length;
    document.getElementById('best').innerText = Math.floor(bestBike.distance);
    
    // Draw Wasted Overlay (Using absolute coordinates, so outside ctx.save/restore of camera)
    if (isHumanMode && humanBike && !humanBike.alive) {
         ctx.save();
         // Reset transform just in case
         ctx.setTransform(1, 0, 0, 1, 0, 0);
         ctx.fillStyle = "rgba(0, 0, 0, 0.5)";
         ctx.fillRect(0, 0, canvas.width, canvas.height);
         ctx.fillStyle = "red";
         ctx.font = "bold 60px Arial";
         ctx.textAlign = "center";
         ctx.fillText("WASTED", canvas.width/2, canvas.height/2);
         ctx.fillStyle = "white";
         ctx.font = "20px Arial";
         ctx.fillText("Respawning...", canvas.width/2, canvas.height/2 + 40);
         ctx.restore();
    }
    
    // Human Mode Respawn Logic Check
    if (isHumanMode) {
        // If human bike is gone or dead, trigger respawn sequence
        if (!humanBike || !humanBike.alive) {
            if (!window.respawnTimeout) {
                 window.respawnTimeout = setTimeout(() => {
                    respawnHuman();
                }, 2000);
            }
        }
    }

    requestAnimationFrame(loop);
}

function nextGeneration() {
    generation++;
    document.getElementById('gen').innerText = generation;
    
    // Sort by fitness
    population.sort((a, b) => b.fitness - a.fitness);
    
    const newPop = [];
    
    // Elitism
    for(let i=0; i<2; i++) {
        const brain = population[i].brain.copy();
        const b = new Bike(150, 200, brain);
        newPop.push(b);
    }
    
    // Selection
    while(newPop.length < POPULATION_SIZE) {
        const p1 = population[Math.floor(Math.random() * 10)]; // Top 10
        const childBrain = p1.brain.copy();
        childBrain.mutate(MUTATION_RATE);
        newPop.push(new Bike(150, 200, childBrain));
    }
    
    // Cleanup old physics bodies
    population.forEach(b => b.removeFromWorld(world));
    
    population = newPop;
    population.forEach(b => b.addToWorld(world));
    
    // Regenerate terrain slightly? Or keep same?
    // Let's keep same for fairness, or regenerate to prevent overfitting.
    // Regenerate for robustness:
    generateTerrain();
}

function resetGeneration() {
    population.forEach(b => b.removeFromWorld(world));
    population = [];
    for(let i=0; i<POPULATION_SIZE; i++) {
        const b = new Bike(150, 200);
        b.addToWorld(world);
        population.push(b);
    }
    generateTerrain();
}

// Start
window.onload = init;
