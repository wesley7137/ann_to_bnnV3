// Establish connection with Socket.IO server
const socket = io('http://localhost:5000');  // Update with your server URL if different

socket.on('connect', function() {
    console.log('Connected to the backend server via WebSocket');
});

socket.on('neural_network_state', function(data) {
    // Call visualization functions with the received data
    visualizeNeurons(data.neuron_states);
    visualizeWeights(data.weights);
    visualizeLastSpikeTime(data.last_spike_times);
    visualizeIonicCurrents(data.ionic_currents);
    // Additional visualization functions can be added here as needed
});

socket.on('disconnect', function() {
    console.log('Disconnected from the backend server');
});


// Function to visualize neurons based on their states
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

// Function to create a neuron visualization
function createNeuronVisualization(index) {
    // Define the geometry and material for the neuron visualization
    let geometry = new THREE.SphereGeometry(1, 32, 32);
    let material = new THREE.MeshBasicMaterial({ color: 0xffffff });
    let neuron = new THREE.Mesh(geometry, material);
    neuron.name = "neuron_" + index;
    return neuron;
}

// Function to update a neuron's visualization based on its state
function updateNeuronVisualization(neuron, state) {
    // Change color, size, or other properties based on the neuron's state
    neuron.material.color.setHex(state.fired ? 0xff0000 : 0x00ff00);
    neuron.scale.setScalar(state.fired ? 1.2 : 1.0);
}

// Function to visualize synaptic weights
function visualizeWeights(weights) {
    // Add your logic here to visualize the weights
    // This could involve updating lines connecting neurons, colors, etc.
}

// Function to fetch the evaluation of the model
async function fetchEvaluateModel() {
    try {
        const response = await fetch('/evaluate_model');
        if (!response.ok) {
            throw new Error(`Network response was not ok: ${response.statusText}`);
        }
        return await response.json();
    } catch (error) {
        console.error('Error fetching evaluation:', error);
        return null; // Or appropriate error handling
    }
}





// Function to visualize neurons based on their states
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

// Function to create a neuron visualization
function createNeuronVisualization(index) {
    // Define the geometry and material for the neuron visualization
    let geometry = new THREE.SphereGeometry(1, 32, 32);
    let material = new THREE.MeshBasicMaterial({ color: 0xffffff });
    let neuron = new THREE.Mesh(geometry, material);
    neuron.name = "neuron_" + index;
    return neuron;
}

// Function to update a neuron's visualization based on its state
function updateNeuronVisualization(neuron, state) {
    // Change color based on whether the neuron has fired or not
    neuron.material.color.setHex(state.fired ? 0xff0000 : 0x00ff00);
    // Adjust the size based on the membrane potential
    neuron.scale.setScalar(1.0 + state.potential / 100);
}

// Function to visualize synaptic weights (connections between neurons)
function visualizeWeights(weights) {
    weights.forEach((weight, index) => {
        // Here, 'weight' would be an object with {from, to, value} properties
        let connection = scene.getObjectByName(`connection_${weight.from}_${weight.to}`);
        if (!connection) {
            connection = createConnectionVisualization(weight.from, weight.to);
            scene.add(connection);
        }
        updateConnectionVisualization(connection, weight.value);
    });
}

function createConnectionVisualization(fromIndex, toIndex) {
    let material = new THREE.LineBasicMaterial({ color: 0xffffff });
    let geometry = new THREE.BufferGeometry();

    // Placeholder for actual neuron positions
    let positions = new Float32Array([0, 0, 0, 1, 1, 1]); // Replace with actual positions
    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));

    let line = new THREE.Line(geometry, material);
    line.name = `connection_${fromIndex}_${toIndex}`;
    return line;
}

function updateConnectionVisualization(connection, weightValue) {
    // Update the thickness of the line based on the weight value
    connection.material.linewidth = Math.abs(weightValue) * 10;

    // Update color based on whether the weight is positive (excitatory) or negative (inhibitory)
    connection.material.color.setHex(weightValue > 0 ? 0x00ff00 : 0xff0000);
}








// Function to update a neuron's visualization based on its membrane potential
function updateNeuronVisualization(neuron, state) {
    // ... (existing code) ...

    // Adjust the intensity of the neuron's color based on the membrane potential
    let intensity = mapMembranePotentialToColorIntensity(state.membrane_potential);
    neuron.material.emissiveIntensity = intensity;
}

// Helper function to map membrane potential to a color intensity value
function mapMembranePotentialToColorIntensity(potential) {
    // Map the potential to a 0-1 range for color intensity (this is arbitrary and can be adjusted)
    return THREE.MathUtils.clamp((potential + 70) / 100, 0, 1);
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

// Extend the neuron visualization update to include Hodgkin-Huxley parameter effects
function updateNeuronVisualization(neuron, state) {
    // ... (existing code) ...

    // Additional visual effects based on Hodgkin-Huxley parameters
    let hhEffect = calculateHHEffect(state.n, state.m, state.h);
    neuron.material.opacity = hhEffect;
}

// Helper function to calculate visual effect based on Hodgkin-Huxley parameters
function calculateHHEffect(n, m, h) {
    // This is a placeholder function; the actual mapping will depend on how you want to visualize these parameters
    // For example, we can make the neuron more transparent if the parameters are below a certain threshold
    return (n + m + h) / 3; // Averaging the parameters for simplicity
}


// Function to visualize ionic currents
function visualizeIonicCurrents(ionicCurrents) {
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




















var gui, gui_info, gui_settings;

function main() {

	var neuralNet = window.neuralNet = new NeuralNetwork();
	scene.add( neuralNet.meshComponents );

	initGui();

	run();

}

// GUI --------------------------------------------------------
/* exported iniGui, updateGuiInfo */

function initGui() {

	gui = new dat.GUI();
	gui.width = 270;

	gui_info = gui.addFolder( 'Info' );
	gui_info.add( neuralNet, 'numNeurons' ).name( 'Neurons' );
	gui_info.add( neuralNet, 'numAxons' ).name( 'Axons' );
	gui_info.add( neuralNet, 'numSignals', 0, neuralNet.settings.limitSignals ).name( 'Signals' );
	gui_info.autoListen = false;

	gui_settings = gui.addFolder( 'Settings' );
	gui_settings.add( neuralNet.settings, 'currentMaxSignals', 0, neuralNet.settings.limitSignals ).name( 'Max Signals' );
	gui_settings.add( neuralNet.particlePool, 'pSize', 0.2, 2 ).name( 'Signal Size' );
	gui_settings.add( neuralNet.settings, 'signalMinSpeed', 0.0, 8.0, 0.01 ).name( 'Signal Min Speed' );
	gui_settings.add( neuralNet.settings, 'signalMaxSpeed', 0.0, 8.0, 0.01 ).name( 'Signal Max Speed' );
	gui_settings.add( neuralNet, 'neuronSizeMultiplier', 0, 2 ).name( 'Neuron Size Mult' );
	gui_settings.add( neuralNet, 'neuronOpacity', 0, 1.0 ).name( 'Neuron Opacity' );
	gui_settings.add( neuralNet, 'axonOpacityMultiplier', 0.0, 5.0 ).name( 'Axon Opacity Mult' );
	gui_settings.addColor( neuralNet.particlePool, 'pColor' ).name( 'Signal Color' );
	gui_settings.addColor( neuralNet, 'neuronColor' ).name( 'Neuron Color' );
	gui_settings.addColor( neuralNet, 'axonColor' ).name( 'Axon Color' );
	gui_settings.addColor( sceneSettings, 'bgColor' ).name( 'Background' );

	gui_info.open();
	gui_settings.open();

	for ( var i = 0; i < gui_settings.__controllers.length; i++ ) {
		gui_settings.__controllers[ i ].onChange( updateNeuralNetworkSettings );
	}

}




function updateNeuralNetworkSettings() {
	neuralNet.updateSettings();
	if ( neuralNet.settings.signalMinSpeed > neuralNet.settings.signalMaxSpeed ) {
		neuralNet.settings.signalMaxSpeed = neuralNet.settings.signalMinSpeed;
		gui_settings.__controllers[ 3 ].updateDisplay();
	}
}

function updateGuiInfo() {
	for ( var i = 0; i < gui_info.__controllers.length; i++ ) {
		gui_info.__controllers[ i ].updateDisplay();
	}
}

// Run --------------------------------------------------------



function update() {
	updateHelpers();

	if (!sceneSettings.pause) {
		var deltaTime = clock.getDelta();
		neuralNet.update(deltaTime);
		updateGuiInfo();
	}
}


// ----  draw loop
function run() {

	requestAnimationFrame( run );
	renderer.setClearColor( sceneSettings.bgColor, 1 );
	renderer.clear();
	update();
	renderer.render( scene, camera );
	stats.update();
	FRAME_COUNT ++;

}

// Events --------------------------------------------------------

window.addEventListener( 'keypress', function ( event ) {

	var key = event.keyCode;

	switch ( key ) {

		case 32:/*space bar*/ sceneSettings.pause = !sceneSettings.pause;
			break;

		case 65:/*A*/
		case 97:/*a*/ sceneSettings.enableGridHelper = !sceneSettings.enableGridHelper;
			break;

		case 83 :/*S*/
		case 115:/*s*/ sceneSettings.enableAxisHelper = !sceneSettings.enableAxisHelper;
			break;

	}

} );


$( function () {
	var timerID;
	$( window ).resize( function () {
		clearTimeout( timerID );
		timerID = setTimeout( function () {
			onWindowResize();
		}, 250 );
	} );
} );


function onWindowResize() {

	WIDTH = window.innerWidth;
	HEIGHT = window.innerHeight;

	pixelRatio = window.devicePixelRatio || 1;
	screenRatio = WIDTH / HEIGHT;

	camera.aspect = screenRatio;
	camera.updateProjectionMatrix();

	renderer.setSize( WIDTH, HEIGHT );
	renderer.setPixelRatio( pixelRatio );

}
