import { OrbitControls } from '/static/js/controls/OrbitControls.js';
import * as THREE from 'three';
// Establish a connection with the Socket.IO server
const socket = io('http://localhost:5000');

// Data structures to hold neuron and synapse objects
let neurons = [];
let synapses = [];

// WebGL context setup
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
const renderer = new THREE.WebGLRenderer();
renderer.setSize(window.innerWidth, window.innerHeight);
document.body.appendChild(renderer.domElement);

// Set camera position
camera.position.z = 50;

// Initialize lighting
const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
scene.add(ambientLight);
const directionalLight = new THREE.DirectionalLight(0xffffff, 1.0);
directionalLight.position.set(1, 1, 1);
scene.add(directionalLight);

// Add orbit controls
const controls = new OrbitControls(camera, renderer.domElement);

// Socket event handlers
socket.on('connect', () => console.log('Connected to the backend server via WebSocket'));
socket.on('neural_network_state', updateNetworkVisualization);
socket.on('disconnect', () => console.log('Disconnected from the backend server'));



num_neurons = 100
input_size = 100  
hidden_size = 100  
output_size = 10 


function updateNetworkVisualization(data) {
    // Update neurons
    neurons.forEach(neuron => scene.remove(neuron));
    neurons = data.neuron_states.map((state, index) => createNeuronVisualization(state, index));

    // Update synapses
    synapses.forEach(synapse => scene.remove(synapse));
    synapses = createSynapses(data.weights);

    // Update visualization of spikes
    visualizeSpikes(data.last_spike_time);

    // Update visualization of ionic currents
    visualizeIonicCurrents(data.I_Na_values, data.I_K_values, data.I_L_values);

    // Update Hodgkin-Huxley parameters visualization
    visualizeHHParameters(data.n_values, data.m_values, data.h_values);

    visualizeNeurons(data.neuron_states);
    visualizeWeights(data.weights);
}

// Additional visualization functions
// Function to visualize neurons with position and activation data
function visualizeNeurons(neuronStates) {
    neuronStates.forEach((state, index) => {
        let neuron = scene.getObjectByName("neuron_" + index);
        if (!neuron) {
            neuron = createNeuronVisualization(index);
            scene.add(neuron);
        }
        updateNeuronVisualization(neuron, state);
    });
}

// Function to visualize synapses based on the weights
function visualizeWeights(weights) {
    weights.forEach((weight, index) => {
        let connection = scene.getObjectByName(`connection_${weight.from}_${weight.to}`);
        if (!connection) {
            connection = createConnectionVisualization(weight.from, weight.to);
            scene.add(connection);
        }
        updateConnectionVisualization(connection, weight.value);
    });
}

function visualizeSpikes(lastSpikeTimes) {
    lastSpikeTimes.forEach((time, index) => {
        // Logic to visualize spikes
        let neuron = neurons[index];
        // Implement spike visualization based on `time`
    });
}

function visualizeIonicCurrents(I_Na_values, I_K_values, I_L_values) {
        ionicCurrents.forEach((current, index) => {
            let neuron = scene.getObjectByName("neuron_" + index);
            if (neuron) {
                // Adjust the neuron's brightness based on ionic currents
                neuron.material.emissiveIntensity = mapCurrentToBrightness(current);
            }
        });
    }
    
    // Helper function to map ionic currents to brightness
    function mapCurrentToBrightness(current) {
        // Map the current to a brightness value (this is arbitrary and can be adjusted)
        return Math.abs(current); // Absolute value for demonstration
    }
    
    // This can be represented by changing the brightness or color of the neurons


function visualizeHHParameters(n_values, m_values, h_values) {
        hhData.forEach((data, index) => {
            let neuron = scene.getObjectByName(`neuron_${index}`);
            if (!neuron) {
                neuron = createNeuronVisualization(index); // You should have a function to create the neuron visualization if it doesn't exist
                scene.add(neuron);
            }
            updateHHVisualization(neuron, data);
        });
    }
    
    // Function to update the visualization based on Hodgkin-Huxley parameters
    function updateHHVisualization(neuron, data) {
        // Example of how you might map the Hodgkin-Huxley parameters to color intensity
        let nIntensity = mapValueToColorIntensity(data.n);
        let mIntensity = mapValueToColorIntensity(data.m);
        let hIntensity = mapValueToColorIntensity(data.h);
        
        // Example of using the parameters to set the neuron color
        neuron.material.color.setRGB(nIntensity, mIntensity, hIntensity);
    }
    
    
    // Helper function to map a Hodgkin-Huxley parameter value to a color intensity
    function mapValueToColorIntensity(value, min, max) {
        // Ensure the value falls within the range [min, max]
        value = Math.min(Math.max(value, min), max);
        // Map the value from [min, max] to [0, 1]
        return (value - min) / (max - min);
    }
    

// Function to create a neuron visualization
function createNeuronVisualization(position) {
    let geometry = new THREE.SphereGeometry(0.5, 32, 32);
    // Ensure that the color property is being set correctly.
    let material = new THREE.MeshPhongMaterial({ color: 0x00ff00 }); // Green color
    let neuron = new THREE.Mesh(geometry, material);
    neuron.position.copy(position);
    neuron.name = `neuron_${position.x}_${position.y}_${position.z}`;
    return neuron;
}
function createSynapses(weights) {
    return weights.map(weight => {
        const material = new THREE.LineBasicMaterial({ color: weight > 0 ? 0x00ff00 : 0xff0000 });
        const points = [neurons[weight.from].position, neurons[weight.to].position];
        const geometry = new THREE.BufferGeometry().setFromPoints(points);
        const synapse = new THREE.Line(geometry, material);
        
        scene.add(synapse);
        return synapse;
    });
}
function calculatePosition(index, totalNeurons) {
    // Determine the layer of the neuron based on the index
    const layers = [inputSize, hiddenSize, outputSize];
    let layerIndex = 0;
    let neuronIndex = index;
    for (let i = 0; i < layers.length; i++) {
        if (neuronIndex >= layers[i]) {
            neuronIndex -= layers[i];
            layerIndex++;
        } else {
            break;
        }
    }

    // Calculate position based on the neuron's layer and index within the layer
    const layerSpacing = 15; // Distance between each layer
    const neuronSpacing = 1.5; // Distance between neurons within a layer
    const layerWidth = Math.sqrt(layers[layerIndex]) * neuronSpacing; // Width of the layer
    const x = (neuronIndex % layerWidth) * neuronSpacing - (layerWidth / 2);
    const y = Math.floor(neuronIndex / layerWidth) * neuronSpacing - (layerWidth / 2);
    const z = layerIndex * layerSpacing;

    return [x, y, z];
}

// Function to visualize the last spike time of neurons
function visualizeLastSpikeTime(lastSpikeTimes) {
    lastSpikeTimes.forEach((lastSpikeTime, index) => {
        let neuron = scene.getObjectByName("neuron_" + index);
        if (neuron) {
            // Trigger a visual effect on the neuron
            triggerSpikeAnimation(neuron, lastSpikeTime);
        }
    });
}

// Helper function to trigger a visual effect on a neuron when it last spiked
function triggerSpikeAnimation(neuron, lastSpikeTime) {
    let currentTime = Date.now();
    let timeSinceLastSpike = currentTime - lastSpikeTime;
    
    // If the last spike was recent, trigger an animation
    if (timeSinceLastSpike < 1000) { // 1000 milliseconds threshold
        neuron.material.emissive.setHex(0xffffff);
    }
}
// Render loop
function animate() {
    requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, camera);
}
animate();

// Handle window resizing
window.addEventListener('resize', onWindowResize, false);
function onWindowResize() {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
}