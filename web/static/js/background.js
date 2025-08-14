// Three.js background animation
let scene, camera, renderer, particles;

function initBackground() {
    scene = new THREE.Scene();
    camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
    renderer = new THREE.WebGLRenderer({ alpha: true });
    
    // Setup renderer
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setPixelRatio(window.devicePixelRatio);
    document.getElementById('bg-animation').appendChild(renderer.domElement);

    // Create particles
    const geometry = new THREE.BufferGeometry();
    const vertices = [];
    const sizes = [];
    const colors = [];

    // Particle colors
    const color1 = new THREE.Color(0x4a90e2); // Primary color
    const color2 = new THREE.Color(0xe74c3c); // Accent color

    // Create random particles
    for (let i = 0; i < 1000; i++) {
        vertices.push(
            Math.random() * 2000 - 1000,
            Math.random() * 2000 - 1000,
            Math.random() * 2000 - 1000
        );

        const mixedColor = color1.clone().lerp(color2, Math.random());
        colors.push(mixedColor.r, mixedColor.g, mixedColor.b);
        sizes.push(Math.random() * 2);
    }

    geometry.setAttribute('position', new THREE.Float32BufferAttribute(vertices, 3));
    geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
    geometry.setAttribute('size', new THREE.Float32BufferAttribute(sizes, 1));

    // Create particle material
    const material = new THREE.PointsMaterial({
        size: 2,
        vertexColors: true,
        transparent: true,
        opacity: 0.8,
        sizeAttenuation: true
    });

    // Create particle system
    particles = new THREE.Points(geometry, material);
    scene.add(particles);

    // Position camera
    camera.position.z = 1000;

    // Start animation
    animate();

    // Handle window resize
    window.addEventListener('resize', onWindowResize, false);
}

function animate() {
    requestAnimationFrame(animate);

    // Rotate particles
    particles.rotation.x += 0.0001;
    particles.rotation.y += 0.0002;

    // Update particle positions based on mouse
    const mouseX = (window.mouseX || 0) * 0.1;
    const mouseY = (window.mouseY || 0) * 0.1;
    particles.rotation.x += (mouseY - particles.rotation.x) * 0.01;
    particles.rotation.y += (mouseX - particles.rotation.y) * 0.01;

    renderer.render(scene, camera);
}

function onWindowResize() {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
}

// Track mouse movement
document.addEventListener('mousemove', (event) => {
    window.mouseX = (event.clientX - window.innerWidth / 2) / 100;
    window.mouseY = (event.clientY - window.innerHeight / 2) / 100;
});

// Initialize background when document is loaded
document.addEventListener('DOMContentLoaded', initBackground);
