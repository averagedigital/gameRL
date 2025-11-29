// Constants
const CHASSIS_WIDTH = 80;
const CHASSIS_HEIGHT = 20;
const WHEEL_RADIUS = 25;

// Socket connection
const socket = io(); // Connects to same host by default if hosted together

const canvas = document.getElementById('world');
const ctx = canvas.getContext('2d');
canvas.width = window.innerWidth;
canvas.height = window.innerHeight;

let terrain = [];
let bikes = [];
let generation = 0;

// LISTEN TO SERVER
socket.on('init', (data) => {
    terrain = data.terrain;
    generation = data.generation;
    document.getElementById('gen').innerText = generation;
});

socket.on('new_generation', (data) => {
    terrain = data.terrain;
    generation = data.generation;
    document.getElementById('gen').innerText = generation;
});

socket.on('state', (data) => {
    bikes = data.bikes;
    draw();
    
    // Update stats
    let alive = bikes.filter(b => b.alive).length;
    document.getElementById('alive').innerText = alive;
    let best = Math.max(...bikes.map(b => b.x));
    document.getElementById('best').innerText = best;
});

function draw() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Camera Logic: Follow leader
    let leader = bikes.reduce((prev, curr) => (curr.alive && curr.x > prev.x) ? curr : prev, bikes[0]);
    if(!leader) leader = bikes[0];
    
    ctx.save();
    let camX = -leader.x + canvas.width / 3;
    ctx.translate(camX, 0);
    
    // Draw Terrain
    ctx.beginPath();
    ctx.strokeStyle = '#333';
    ctx.lineWidth = 5;
    if(terrain.length > 0) {
        ctx.moveTo(terrain[0].x, terrain[0].y);
        for(let i=1; i<terrain.length; i++) {
            ctx.lineTo(terrain[i].x, terrain[i].y);
        }
    }
    ctx.stroke();
    
    // Draw Bikes
    bikes.forEach(b => {
        if(!b.alive) return;
        
        // Chassis
        ctx.save();
        ctx.translate(b.x, b.y);
        ctx.rotate(b.a);
        ctx.fillStyle = '#e74c3c';
        ctx.fillRect(-CHASSIS_WIDTH/2, -CHASSIS_HEIGHT/2, CHASSIS_WIDTH, CHASSIS_HEIGHT);
        ctx.restore();
        
        // Wheels
        [ {x: b.wb_x, y: b.wb_y, a: b.wb_a}, {x: b.wf_x, y: b.wf_y, a: b.wf_a} ].forEach(w => {
            ctx.beginPath();
            ctx.arc(w.x, w.y, WHEEL_RADIUS, 0, Math.PI * 2);
            ctx.fillStyle = '#34495e';
            ctx.fill();
            // Spoke
            ctx.beginPath();
            ctx.moveTo(w.x, w.y);
            ctx.lineTo(w.x + Math.cos(w.a)*WHEEL_RADIUS, w.y + Math.sin(w.a)*WHEEL_RADIUS);
            ctx.strokeStyle = '#fff';
            ctx.stroke();
        });
        
        // Rays
        if(b.rays) {
            b.rays.forEach(r => {
                ctx.beginPath();
                ctx.moveTo(r.start.x, r.start.y);
                ctx.lineTo(r.end.x, r.end.y);
                ctx.strokeStyle = r.hit ? 'rgba(255, 0, 0, 0.5)' : 'rgba(0, 255, 0, 0.2)';
                ctx.lineWidth = 1;
                ctx.stroke();
            });
        }
    });
    
    ctx.restore();
}

// Handle resize
window.onresize = () => {
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
};
