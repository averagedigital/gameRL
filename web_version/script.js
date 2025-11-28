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
const MOTOR_SPEED = 0.5; // Base speed, human will have multiplier
const LEAN_SPEED = 0.1;

// GA Config
const POPULATION_SIZE = 12; // Increased from 2 for better learning
const MUTATION_RATE = 0.1; // Back to standard rate
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
    
    // Simple Backpropagation (SGD)
    train(inputs, targets, learningRate = 0.1) {
        // Forward pass (we need intermediate values)
        // Hidden Layer
        let h1_in = this.matmul([inputs], this.w1);
        let h1 = this.addBias(h1_in, this.b1);
        let h1_act = this.tanh(h1);
        
        // Output Layer
        let out_in = this.matmul(h1_act, this.w2);
        let out = this.addBias(out_in, this.b2);
        let out_act = this.tanh(out);
        
        // Error (MSE derivative is just error for linear, for tanh it's error * derivative)
        // Target is [gas, lean]
        let error = [targets[0] - out_act[0][0], targets[1] - out_act[0][1]];
        
        // Backward Pass
        // Output Gradients
        // dE/dOut * dOut/dNet = error * (1 - out^2)
        let out_grad = [];
        for(let i=0; i<this.outputSize; i++) {
            out_grad[i] = error[i] * (1 - out_act[0][i] * out_act[0][i]);
        }
        
        // Hidden Gradients
        // dE/dH = (dE/dOut * dOut/dH) * dH/dNet
        let h_grad = [];
        for(let i=0; i<this.hiddenSize; i++) {
            let error_sum = 0;
            for(let j=0; j<this.outputSize; j++) {
                error_sum += out_grad[j] * this.w2[i][j];
            }
            h_grad[i] = error_sum * (1 - h1_act[0][i] * h1_act[0][i]);
        }
        
        // Update Weights
        // W2 += H.T * Out_Grad * LR
        for(let i=0; i<this.hiddenSize; i++) {
            for(let j=0; j<this.outputSize; j++) {
                this.w2[i][j] += h1_act[0][i] * out_grad[j] * learningRate;
            }
        }
        // B2 += Out_Grad * LR
        for(let i=0; i<this.outputSize; i++) {
            this.b2[0][i] += out_grad[i] * learningRate;
        }
        
        // W1 += In.T * H_Grad * LR
        for(let i=0; i<this.inputSize; i++) {
            for(let j=0; j<this.hiddenSize; j++) {
                this.w1[i][j] += inputs[i] * h_grad[j] * learningRate;
            }
        }
        // B1 += H_Grad * LR
        for(let i=0; i<this.hiddenSize; i++) {
            this.b1[0][i] += h_grad[i] * learningRate;
        }
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
        
        // Collision Filter Logic for Ghost Mode:
        // We want:
        // 1. Bike parts (chassis, wheels) colliding with TERRAIN.
        // 2. Bike parts NOT colliding with OTHER bikes.
        // 3. Bike parts NOT colliding with EACH OTHER (self-collision).
        
        // Strategy:
        // Use 'category' and 'mask' for selective collision.
        // Use 'group' for self-collision exemption.
        
        const CATEGORY_BIKE = 0x0002;
        const CATEGORY_TERRAIN = 0x0001;
        
        // Each bike instance gets a UNIQUE negative group to prevent self-collision.
        // However, if we want to prevent BIKE-BIKE collision, we must ensure their MASKS do not include CATEGORY_BIKE.
        
        // Problem: Matter.js 'group' overrides category/mask if non-zero.
        // "If the two bodies have the same non-zero value for collisionFilter.group, they will always collide if the value is positive, and they will never collide if the value is negative."
        // "If the two bodies have different values for collisionFilter.group, or if one of them is zero, then the collisionFilter.category and collisionFilter.mask rules are used."
        
        // So for GHOST MODE (no bike-bike collision):
        // 1. Give every bike a UNIQUE negative group? No, that only stops self-collision.
        // 2. If we give ALL bikes the SAME negative group, they won't collide with each other!
        //    BUT then wheels won't collide with chassis? Yes, parts of same bike won't collide.
        //    AND parts of DIFFERENT bikes won't collide.
        //    PERFECT for Ghost Mode!
        
        // Wait, if all have same negative group, they don't collide with terrain?
        // No, terrain usually has group 0 (default).
        // "If different values... category/mask used."
        // So Bike (Group -1) vs Terrain (Group 0) -> Check Mask.
        
        // Solution:
        // ALL bikes get the SAME negative group constant.
        // This disables collision between ANY bike parts (self or other).
        
        const GHOST_GROUP = -1; // Constant for all bikes
        
        const filter = {
            group: GHOST_GROUP,
            category: CATEGORY_BIKE,
            mask: CATEGORY_TERRAIN // Only collide with terrain (which is default category 1)
        };

        // Chassis
        this.chassis = Bodies.rectangle(x, y, CHASSIS_WIDTH, CHASSIS_HEIGHT, { 
            collisionFilter: filter,
            density: 0.04,
            friction: 0.5,
            label: 'chassis'
        });
        
        // Wheels
        this.wheelBack = Bodies.circle(x - 30, y + 20, WHEEL_RADIUS, { 
            collisionFilter: filter,
            friction: 0.9,
            density: 0.05,
            restitution: 0.2
        });
        
        this.wheelFront = Bodies.circle(x + 30, y + 20, WHEEL_RADIUS, { 
            collisionFilter: filter,
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
            // Human Control - Standard WASD / Arrows
            // W / Up = Gas (Forward)
            // S / Down = Brake (Reverse)
            // A / Left = Lean Back (Rotate CCW)
            // D / Right = Lean Forward (Rotate CW)
            
            if (keys['ArrowUp'] || keys['KeyW']) gas = 1;
            if (keys['ArrowDown'] || keys['KeyS']) gas = -1;
            if (keys['ArrowRight'] || keys['KeyD']) lean = 1;   // Lean forward
            if (keys['ArrowLeft'] || keys['KeyA']) lean = -1; // Lean back
        } else {
            // AI Control
            // --- INPUT NORMALIZATION (ML Engineering) ---
            // Neural Networks love numbers between -1 and 1.
            
            // 1. Angle: -PI to PI -> -1 to 1
            const angle = this.chassis.angle / Math.PI; 
            
            // 2. Angular Velocity: Usually -0.2 to 0.2. Scale by 5.
            const angVel = Math.max(-1, Math.min(1, this.chassis.angularVelocity * 5));
            
            // 3. Velocity X: Max speed ~20. Scale by 1/20.
            const vX = Math.max(-1, Math.min(1, this.chassis.velocity.x / 20));
            
            // 4. Velocity Y: Jump speed ~10. Scale by 1/10.
            const vY = Math.max(-1, Math.min(1, this.chassis.velocity.y / 10));
            
            // 5. Height: Relative to ground (0-1). 
            // We use absolute Y, but better is raycast center?
            // Let's keep absolute normalized for now.
            const height = (600 - this.chassis.position.y) / 600;
            
            const inputs = [angle, angVel, vX, vY, height, ...rays];
            
            // Record training data if human
            if (this.isHuman) {
                 trainingData.push({
                     inputs: inputs,
                     targets: [gas, lean]
                 });
            } else {
                 const output = this.brain.predict(inputs);
                 gas = output[0]; 
                 lean = output[1];
            }
        }
        
        // Apply Motor
        // We apply torque directly to wheel or AngularVelocity
        if (gas !== 0) {
            // Forward (negative torque/velocity in matter.js usually? depends on coordinate)
            // Matter.js clockwise is positive. So forward is clockwise?
            // Let's try setting angular velocity target
            let speed = MOTOR_SPEED;
            if(this.isHuman) speed *= 2; // Human needs more power
            Body.setAngularVelocity(this.wheelBack, this.wheelBack.angularVelocity + gas * speed);
        } else {
             // BRAKING / FRICTION when no gas
             // If human and no input, dampen wheel spin
             if (this.isHuman) {
                 Body.setAngularVelocity(this.wheelBack, this.wheelBack.angularVelocity * 0.9);
             }
        }
        
        // Detect if airborne (simple check: wheels not colliding?)
        // Matter.js collision check is expensive to do properly every frame without events.
        // Heuristic: check if wheels have very low Y velocity change or vertical support?
        // Easier: Just give "Air Control" always, physics will dampen it on ground.
        // But we want EXTRA power in air.
        
        // Let's just boost the lean power globally for humans, it's fun.
        
        // Apply Lean Torque (Rotation)
        // Reduced power for smoother control (was 0.2, causing instant flips)
        let leanPower = 0.05;
        if(this.isHuman) leanPower = 0.08; // Just enough to rotate, but not flip instantly
        
        // "Space Control" - Air Strafe?
        // If user wants "Left/Right" control in space, maybe they mean moving the bike horizontally in air?
        // Let's add a small horizontal force for A/D in addition to rotation.
        if (this.isHuman && lean !== 0) {
            // Apply slight horizontal push
            Body.applyForce(this.chassis, this.chassis.position, {x: lean * 0.002, y: 0});
        }

        Body.setAngularVelocity(this.chassis, this.chassis.angularVelocity + lean * leanPower);
        
        this.distance = this.chassis.position.x;
        this.fitness = this.distance;
        
        // --- STRICT DEATH CONDITIONS (ML Engineering) ---
        
        // 1. Head Hit / Upside Down
        // Check collision of chassis with terrain? 
        // Or simple angle check: If angle > 90 degrees (PI/2), you are dead.
        // PI/2 = 1.57 radians. 
        if (Math.abs(this.chassis.angle) > Math.PI / 2) {
             this.alive = false;
        }
        
        // 2. Idle / Stuck Check
        // If speed is very low for too long -> Die.
        // Velocity threshold: 0.1
        // Time threshold: 180 frames (3 seconds)
        const velocity = Math.sqrt(this.chassis.velocity.x**2 + this.chassis.velocity.y**2);
        if (velocity < 0.5) {
            this.idleTime = (this.idleTime || 0) + 1;
            if (this.idleTime > 180) {
                this.alive = false; // You are boring, die.
            }
        } else {
            this.idleTime = 0;
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
let trainingData = []; // Store (inputs, targets)
let frameCount = 0;
let maxFrames = 60 * 120; // 120 seconds max per gen (increased from 30)

function init() {
    // Global Error Handler to catch "Freezes"
    window.onerror = function(msg, url, line, col, error) {
        alert("Game Error: " + msg + "\nLine: " + line);
        // Force reset
        isHumanMode = false;
        resetGeneration();
        return false;
    };

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
    
    // Create Population (Spawn one by one)
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
    
    // Debugging and Robust Event Listeners
    const exportBtn = document.getElementById('export-btn');
    if(exportBtn) {
        exportBtn.onclick = () => {
            console.log("Export button clicked");
            exportBestBrain();
        };
    } else { console.error("Export button not found!"); }

    const importBtn = document.getElementById('import-btn');
    if(importBtn) {
        importBtn.onclick = () => {
            console.log("Import button clicked");
            importBrain();
        };
    } else { console.error("Import button not found!"); }
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
    window.respawnTimeout = null;
    
    // TRAIN ON DEATH
    if (trainingData.length > 0 && humanBike) {
        // Train the brain on the collected data
        console.log("Training on " + trainingData.length + " frames...");
        // Multiple epochs for better learning
        for(let epoch=0; epoch<5; epoch++) {
            for(let i=0; i<trainingData.length; i++) {
                humanBike.brain.train(trainingData[i].inputs, trainingData[i].targets, 0.05);
            }
        }
        console.log("Training complete!");
        // Keep this brain for the next respawn?
        // Ideally we want to export THIS brain.
        // For now, we respawn a new bike with THIS brain.
    }
    const learnedBrain = (humanBike && trainingData.length > 0) ? humanBike.brain : null;
    trainingData = []; // Clear data
    
    // Clear ALL physics bodies related to bikes
    if (humanBike) {
        humanBike.removeFromWorld(world);
    }
    population.forEach(b => b.removeFromWorld(world));
    population = [];
    
    // Create fresh bike with learned brain or new one
    humanBike = new Bike(150, 200, learnedBrain, true);
    humanBike.addToWorld(world);
    population = [humanBike];
    
    // Force physics update
    Engine.update(engine, 1000/60);
}

function exportBestBrain() {
    console.log("Exporting...");
    // Find best AI from previous or current gen
    // Since we constantly mutate, let's grab the current survivor with max distance
    // Or ideally, we should store the 'champion' separately.
    // For now, let's grab the leader.
    if(isHumanMode) {
        // If human mode, export HUMAN brain
        if (humanBike) {
             const data = JSON.stringify(humanBike.brain);
             showModal("Copy Human Brain:", data);
            return;
        }
    }
    
    if (population.length === 0) {
        alert("No brains to copy!");
        return;
    }
    
    const best = population.reduce((prev, current) => (prev.distance > current.distance) ? prev : current);
    const data = JSON.stringify(best.brain);
    
    // Custom Modal Output instead of clipboard (more reliable)
    showModal("Copy this code:", data);
}

function importBrain() {
    // Custom Modal Input
    const data = prompt("Paste the Brain code here:");
    if(!data) return;
    
    try {
        const json = JSON.parse(data);
        // Create a new population based on this brain
        resetGeneration();
        
        // Setup new generation
        // Critical: We want this brain to be the BASE for everyone.
        
        // 1. Create the champion template
        const champion = new NeuralNetwork(5 + RAY_COUNT, 8, 2);
        champion.w1 = json.w1; champion.b1 = json.b1; 
        champion.w2 = json.w2; champion.b2 = json.b2;
        
        const newBrains = [];
        
        // 2. Clone it for everyone!
        // First one is exact copy (Elitism)
        newBrains.push(champion.copy());
        
        // Rest are MUTATED versions so evolution continues
        // If we just copy exact, they will all do the same thing and die together.
        // We want to "continue evolution from this save point".
        for(let i=1; i<POPULATION_SIZE; i++) {
            const clone = champion.copy();
            clone.mutate(MUTATION_RATE);
            newBrains.push(clone);
        }
        
        // Force inject these brains into the next generation spawn logic
        window.nextGenBrains = newBrains;
        
        // Kill current population and respawn immediately with new brains
        population.forEach(b => b.removeFromWorld(world));
        population = [];
        
        createPopulation();
        
        // Don't regenerate terrain? Or do? 
        // Maybe keep terrain so user can see if it beats the current level.
        // But usually fairness requires new terrain. Let's keep it consistent.
        generateTerrain(); 
        
        alert("Brain imported successfully! The population has been replaced.");
        
    } catch(e) {
        alert("Invalid Brain string! Check console.");
        console.error(e);
    }
}

function showModal(title, text) {
    // Simple overlay
    let overlay = document.getElementById('modal-overlay');
    if(!overlay) {
        overlay = document.createElement('div');
        overlay.id = 'modal-overlay';
        overlay.style.cssText = 'position:fixed;top:0;left:0;width:100%;height:100%;background:rgba(0,0,0,0.5);display:flex;justify-content:center;align-items:center;z-index:1000;';
        document.body.appendChild(overlay);
    }
    
    overlay.innerHTML = `
        <div style="background:white;padding:20px;border-radius:10px;width:80%;max-width:500px;text-align:center;">
            <h3>${title}</h3>
            <textarea id="modal-text" style="width:100%;height:150px;margin:10px 0;">${text}</textarea>
            <br>
            <button onclick="document.getElementById('modal-text').select();document.execCommand('copy');">Copy to Clipboard</button>
            <button onclick="document.getElementById('modal-overlay').remove()">Close</button>
        </div>
    `;
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
    // If we have existing brains to carry over (from nextGeneration), use them
    // Otherwise create new ones
    // We want to spawn them staggered
    
    // Clear array logic handled by caller, but we need to ensure clean state
    
    // Helper to spawn one bike
    // ... removed complex staggered logic ...

    // Solution: Spawn all at once but with X spacing.
    // Bike 1: x=150
    // Bike 2: x=100
    // Bike 3: x=50...
    
    // Ghost Mode is ON: Spawn all at SAME spot (150)
    // Y Position: 200 is too low? Terrain starts at Y=500 (screen coords, where 0 is top).
    // Let's spawn them HIGH above the runway to guarantee drop.
    // Runway is at Y=500. Spawn at Y=300 (above).
    for(let i=0; i<POPULATION_SIZE; i++) {
        const brain = (window.nextGenBrains && window.nextGenBrains[i]) ? window.nextGenBrains[i] : null;
        // Spacing: 0 (Ghost Mode)
        // Spawn Y: 300 (Safety Drop)
        const b = new Bike(150, 300, brain);
        b.addToWorld(world);
        population.push(b);
    }
    window.nextGenBrains = null; // Clear buffer
}

function nextGeneration() {
    generation++;
    document.getElementById('gen').innerText = generation;
    
    // Sort by fitness
    population.sort((a, b) => b.fitness - a.fitness);
    
    const newBrains = [];
    
    // Elitism (Top 2)
    for(let i=0; i<2; i++) {
        if(population[i]) newBrains.push(population[i].brain.copy());
    }
    
    // Selection
    while(newBrains.length < POPULATION_SIZE) {
        // Tournament
        let p1 = population[Math.floor(Math.random() * population.length)];
        let p2 = population[Math.floor(Math.random() * population.length)];
        // Safety check
        if(!p1) p1 = population[0];
        if(!p2) p2 = population[0];
        
        let parent = (p1.fitness > p2.fitness) ? p1 : p2;
        
        const childBrain = parent.brain.copy();
        childBrain.mutate(MUTATION_RATE);
        newBrains.push(childBrain);
    }
    
    // Store brains for next spawn
    window.nextGenBrains = newBrains;
    
    // Cleanup old physics bodies
    population.forEach(b => b.removeFromWorld(world));
    population = [];
    
    createPopulation();
    
    // Regenerate terrain
    generateTerrain();
}

function resetGeneration() {
    population.forEach(b => b.removeFromWorld(world));
    population = [];
    window.nextGenBrains = null;
    createPopulation();
    generateTerrain();
}

function generateTerrain() {
    if (terrainBody) Composite.remove(world, terrainBody);
    
    let x = 0;
    let y = 500;
    const points = [{x: 0, y: 1000}]; // Bottom left anchor
    points.push({x: 0, y: y}); // Start
    
    window.terrainVertices = []; // Store for raycasting
    
    // SAFE ZONE: First 20 segments (1000px) are FLAT (Incubator)
    for(let i=0; i<300; i++) {
        window.terrainVertices.push({x, y});
        x += 50;
        
        // Only start randomizing after safe zone
        if (i > 20) {
            // Curriculum Learning: Make it harder as we go?
            // Or just consistent hills.
            y += (Math.random() - 0.5) * 60;
            if (y > 800) y = 800;
            if (y < 200) y = 200;
            
            // Add hill
            if (i > 30 && i % 20 < 10) y -= 40;
        }
        
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
    try {
        _loopContent();
    } catch (e) {
        console.error("Game Loop Crashed:", e);
        // Try to recover
        requestAnimationFrame(loop);
    }
}

function _loopContent() {
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
        
        // Only progress frame count if not human
        if (!isHumanMode) {
             frameCount++;
             
             // Wait for at least 60 frames (1 sec) before checking death to allow spawn to settle
             if (frameCount > 60) {
                 // Restart only if ALL are dead OR time is up
                 if (aliveCount === 0 || frameCount > maxFrames) {
                    nextGeneration();
                    break;
                }
             }
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
    frameCount = 0; // Reset timer
    
    // Sort by fitness
    population.sort((a, b) => b.fitness - a.fitness);
    
    const newBrains = [];
    
    // Elitism (Top 2)
    for(let i=0; i<2; i++) {
        if(population[i]) newBrains.push(population[i].brain.copy());
    }
    
    // Selection
    while(newBrains.length < POPULATION_SIZE) {
        // Tournament
        let p1 = population[Math.floor(Math.random() * population.length)];
        let p2 = population[Math.floor(Math.random() * population.length)];
        // Safety check
        if(!p1) p1 = population[0];
        if(!p2) p2 = population[0];
        
        let parent = (p1.fitness > p2.fitness) ? p1 : p2;
        
        const childBrain = parent.brain.copy();
        childBrain.mutate(MUTATION_RATE);
        newBrains.push(childBrain);
    }
    
    // Store brains for next spawn
    window.nextGenBrains = newBrains;
    
    // Cleanup old physics bodies
    population.forEach(b => b.removeFromWorld(world));
    population = [];
    
    createPopulation();
    
    // Regenerate terrain
    generateTerrain();
}

function resetGeneration() {
    population.forEach(b => b.removeFromWorld(world));
    population = [];
    window.nextGenBrains = null;
    createPopulation();
    generateTerrain();
}

// Start
window.onload = init;
