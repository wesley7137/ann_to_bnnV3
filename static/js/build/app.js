// Assuming the data structure provided by the backend
let neurons = [], synapses = [];

// WebSocket setup and event handling
const socket = io('http://localhost:5000');
socket.on('connect', () => console.log('Connected to backend via WebSocket'));
socket.on('neural_network_state', data => {
    updateNeuronsAndSynapses(data);
});

socket.on('neural_network_state', function(data) {
    console.log('Received neural network state:', data);
    if (data.hasOwnProperty('neuron_states') && data.hasOwnProperty('weights')) {
        visualizeNeurons(data.neuron_states);
        visualizeWeights(data.weights);
        visualizeLastSpikeTime(data.last_spike_times);
        ionic_currents=(data.I_Na_values, data.I_K_values, data.I_L_values);
        visualizeIonicCurrents(data.ionic_currents);
        visualizeBiases(data.biases);
        hh_parameters=(data.n_values, data.m_values, data.h_values); 
        visualizeHHParameters(data.hh_parameters);

    } else {
        console.error('Received data is missing required properties:', data);
    }
});



// Scene, Camera, and Renderer setup
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(75, window.innerWidth/window.innerHeight, 0.1, 1000);
const renderer = new THREE.WebGLRenderer();
renderer.setSize(window.innerWidth, window.innerHeight);
document.body.appendChild(renderer.domElement);

// Camera positioning for overview of the network
camera.position.z = 50;

// Lighting
const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
scene.add(ambientLight);

const directionalLight = new THREE.DirectionalLight(0xffffff, 1);
directionalLight.position.set(0, 1, 1);
scene.add(directionalLight);
// Function to create a neuron visualization at a given index
function createNeuronVisualization(index) {
    const geometry = new THREE.SphereGeometry(1, 32, 32);
    const material = new THREE.MeshStandardMaterial({ color: 0xffffff });
    const neuron = new THREE.Mesh(geometry, material);
    neuron.name = "neuron_" + index;
    // Positioning logic here (e.g., based on a grid or other structure)
    // neuron.position.set(x, y, z);
    return neuron;
}

// Function to create a visualization of a connection between two neurons
function createConnectionVisualization(fromIndex, toIndex) {
    const material = new THREE.LineBasicMaterial({ color: 0xffffff });
    const geometry = new THREE.BufferGeometry();
    // Add positioning logic here based on neuron positions
    const positions = new Float32Array(6); // Placeholder for actual positions
    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    const line = new THREE.Line(geometry, material);
    line.name = `connection_${fromIndex}_${toIndex}`;
    return line;
}

// Function to update neuron visualization based on its state
function updateNeuronVisualization(neuron, state) {
    neuron.material.color.setHex(state.fired ? 0xff0000 : 0x00ff00);
    neuron.scale.setScalar(state.fired ? 1.2 : 1.0);
}

// Function to update connection visualization based on the synaptic weight
function updateConnectionVisualization(connection, weightValue) {
    // Modify line thickness based on weightValue; Three.js does not support line thickness changes in WebGL
    // As an alternative, consider using cylinder geometries or a shader for visualizing different thicknesses
    connection.material.color.setHex(weightValue > 0 ? 0x00ff00 : 0xff0000);
}
// Animation loop
function animate() {
    requestAnimationFrame(animate);
    // Additional updates like controls and stats here
    renderer.render(scene, camera);
}

// Start animation loop
animate();

// Event listener for window resize to adjust camera and renderer
window.addEventListener('resize', onWindowResize, false);
function onWindowResize() {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
}




function updateNeuronsAndSynapses(data) {
    updateNeuronStates(data.neuron_states, data.membrane_potentials);
    updateSynapticWeights(data.weights);
    // Other updates like spike times, currents, etc.
    // ...
}
// Updates the neuron states
function updateNeuronStates(neuronStates, membranePotentials) {
    neuronStates.forEach((state, index) => {
        let neuron = neurons[index];
        if (!neuron) {
            neuron = createNeuronVisualization(index);
            neurons[index] = neuron;
            scene.add(neuron);
        }
        updateNeuronVisualization(neuron, state, membranePotentials[index]);
    });
}

// Updates the synaptic weights
function updateSynapticWeights(weights) {
    // Clear existing synapses
    synapses.forEach(synapse => scene.remove(synapse));
    synapses = [];

    weights.forEach((weightData, index) => {
        const fromNeuron = neurons[weightData.from];
        const toNeuron = neurons[weightData.to];
        const synapse = createSynapseVisualization(fromNeuron, toNeuron, weightData.value);
        synapses.push(synapse);
        scene.add(synapse);
    });
}
// Creates a neuron visualization based on its index and position
function createNeuronVisualization(index) {
    // Determine the position based on the index or some layout algorithm
    const position = calculateNeuronPosition(index);
    const geometry = new THREE.SphereGeometry(0.5, 32, 32);
    const material = new THREE.MeshStandardMaterial({ color: 0xffffff });
    const neuron = new THREE.Mesh(geometry, material);
    neuron.position.copy(position);
    return neuron;
}

// Creates a synapse visualization between two neurons
function createSynapseVisualization(fromNeuron, toNeuron, weight) {
    const material = new THREE.LineBasicMaterial({ color: weight > 0 ? 0x00ff00 : 0xff0000 });
    const geometry = new THREE.BufferGeometry().setFromPoints([fromNeuron.position, toNeuron.position]);
    const synapse = new THREE.Line(geometry, material);
    return synapse;
}
// Updates a neuron's visualization based on its state and potential
function updateNeuronVisualization(neuron, state, potential) {
    const firedColor = 0xff0000; // Red for fired
    const restColor = 0x00ff00; // Green for resting
    neuron.material.color.setHex(state.fired ? firedColor : restColor);
    neuron.scale.setScalar(1.0 + potential / 100); // Adjust the scale based on potential
}

// Updates a synapse visualization based on the synaptic weight
function updateSynapseVisualization(synapse, weight) {
    // Here you might adjust the synapse color or other properties based on the weight
    synapse.material.color.setHex(weight > 0 ? 0x00ff00 : 0xff0000);
    // Note: Thickness cannot be changed in Three.js, consider other approaches if needed
}
// Animation loop
function animate() {
    requestAnimationFrame(animate);
    TWEEN.update(); // If using tween.js for smooth transitions
    renderer.render(scene, camera);
}

// Start the animation loop
animate();
// Layout algorithm to calculate the position of neurons
function calculateNeuronPosition(index) {
    // Implement a layout algorithm based on the index
    // Example: Arrange neurons in a grid or circular layout
    // This is just a placeholder to illustrate, replace with actual logic
    const x = index % 10 - 5;
    const y = Math.floor(index / 10) - 5;
    const z = 0;
    return new THREE.Vector3(x, y, z);
}
