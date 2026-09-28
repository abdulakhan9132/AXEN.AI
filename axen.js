/**
 * AXEN AI — A Self-Contained Educational Neural Network
 *
 * This project demonstrates how an artificial neural network learns the XOR
 * logic problem from scratch using forward propagation, backpropagation,
 * and gradient descent.
 *
 * Architecture:
 *   2 Inputs  -->  4 Hidden Neurons (Sigmoid)  -->  1 Output Neuron (Sigmoid)
 *
 * No external machine learning libraries or APIs are used.
 */

const fs = require("fs");
const path = require("path");
const readline = require("readline");

// ==================================================
// DATASET
// ==================================================
// The truth table for XOR (Exclusive OR):
// 0 XOR 0 = 0
// 0 XOR 1 = 1
// 1 XOR 0 = 1
// 1 XOR 1 = 0
// These targets exist ONLY here in the dataset.
const dataset = [
    { input: [0, 0], target: 0 },
    { input: [0, 1], target: 1 },
    { input: [1, 0], target: 1 },
    { input: [1, 1], target: 0 }
];

// Configuration constants
const LEARNING_RATE = 0.5;           // How big of a step we take down the error gradient
const MAX_TRAINING_STEPS = 100000;    // Safety limit to prevent infinite loops
const TARGET_ERROR = 0.01;           // Training stops when average dataset squared error < 0.01
const BRAIN_FILE = path.join(__dirname, "axen_brain.json");

// ==================================================
// ACTIVATION FUNCTIONS
// ==================================================

/**
 * The Sigmoid activation function.
 * Squeezes any real number into a smooth curve between 0 and 1.
 * Formula: S(x) = 1 / (1 + e^(-x))
 */
function sigmoid(x) {
    return 1 / (1 + Math.exp(-x));
}

/**
 * The derivative of the Sigmoid function.
 * Tells us the slope (rate of change) of the sigmoid curve.
 * Note: 'output' is assumed to already be the result of sigmoid(x).
 * Formula: S'(x) = S(x) * (1 - S(x)) = output * (1 - output)
 */
function sigmoidDerivative(output) {
    return output * (1 - output);
}

// ==================================================
// BRAIN INITIALIZATION
// ==================================================

/**
 * Generates a small random weight between -1.0 and +1.0.
 * Random weights break symmetry, allowing different hidden neurons
 * to learn different features of the problem.
 */
function randomWeight() {
    return (Math.random() * 2) - 1;
}

/**
 * Creates a brand new brain structure with random weights and biases.
 *
 * Weights representation:
 * - weightsInputHidden: 2 inputs × 4 hidden neurons (array of 2 arrays, each of length 4)
 *   weightsInputHidden[i][h] = weight connecting input i to hidden neuron h
 * - biasHidden: array of 4 biases, one for each hidden neuron
 * - weightsHiddenOutput: 4 hidden neurons × 1 output neuron (array of 4 arrays, each of length 1)
 *   weightsHiddenOutput[h][0] = weight connecting hidden neuron h to output neuron 0
 * - biasOutput: array of length 1, holding the bias for output neuron 0
 */
function createBrain() {
    return {
        // 2 inputs × 4 hidden neurons
        weightsInputHidden: [
            [randomWeight(), randomWeight(), randomWeight(), randomWeight()], // weights from input 0
            [randomWeight(), randomWeight(), randomWeight(), randomWeight()]  // weights from input 1
        ],
        // 4 hidden neuron biases
        biasHidden: [
            randomWeight(),
            randomWeight(),
            randomWeight(),
            randomWeight()
        ],
        // 4 hidden neurons × 1 output neuron
        weightsHiddenOutput: [
            [randomWeight()], // hidden neuron 0 -> output
            [randomWeight()], // hidden neuron 1 -> output
            [randomWeight()], // hidden neuron 2 -> output
            [randomWeight()]  // hidden neuron 3 -> output
        ],
        // 1 output bias
        biasOutput: [
            randomWeight()
        ]
    };
}

// ==================================================
// FORWARD PASS
// ==================================================

/**
 * Calculates predictions by passing inputs forward through the network.
 *
 * Steps:
 * 1. For each hidden neuron, compute the weighted sum of inputs plus hidden bias.
 * 2. Apply the sigmoid activation to obtain hidden activations.
 * 3. For the output neuron, compute the weighted sum of hidden activations plus output bias.
 * 4. Apply the sigmoid activation to obtain the final output.
 * 5. Return both hidden activations and final output.
 */
function forward(brain, input) {
    // 1 & 2. Calculate hidden-layer weighted sums and apply sigmoid
    const hidden = [];
    for (let h = 0; h < 4; h++) {
        // Start with the hidden neuron's bias
        let sum = brain.biasHidden[h];
        // Add weighted contributions from each input
        for (let i = 0; i < 2; i++) {
            sum += input[i] * brain.weightsInputHidden[i][h];
        }
        // Squeeze sum into (0, 1) using sigmoid
        hidden.push(sigmoid(sum));
    }

    // 3 & 4. Calculate output weighted sum and apply sigmoid
    let sumOutput = brain.biasOutput[0];
    for (let h = 0; h < 4; h++) {
        sumOutput += hidden[h] * brain.weightsHiddenOutput[h][0];
    }
    const output = sigmoid(sumOutput);

    // Return the intermediate hidden values and the final output
    return {
        hidden: hidden,
        output: output
    };
}

// ==================================================
// BACKPROPAGATION
// ==================================================

/**
 * Performs backpropagation to compute error gradients and update weights and biases.
 *
 * Step-by-step logic:
 * 1. Output Error: (target - output). How far off was the prediction?
 * 2. Output Delta: Error scaled by the sigmoid derivative at the output.
 * 3. Hidden Error: How much did each hidden neuron contribute to the output error?
 *    IMPORTANT: We compute this using the OLD hidden->output weights BEFORE updating them.
 * 4. Hidden Deltas: Each hidden error scaled by that neuron's sigmoid derivative.
 * 5. Update Hidden -> Output weights: Move weights in direction of gradient.
 * 6. Update Output bias: Shift threshold for the output neuron.
 * 7. Update Input -> Hidden weights: Adjust connections between inputs and hidden neurons.
 * 8. Update Hidden biases: Shift thresholds for hidden neurons.
 *
 * Returns the squared error for this example.
 */
function backpropagate(brain, input, target, result) {
    // Step 1: Calculate output error (target minus our prediction)
    const outputError = target - result.output;

    // Step 2: Calculate output delta
    // By chain rule: dError/dSum = (target - output) * sigmoidDerivative(output)
    const outputDelta = outputError * sigmoidDerivative(result.output);

    // Step 3: Calculate hidden-layer error
    // Each hidden neuron contributed through weightsHiddenOutput[h][0].
    // We MUST use the old weights before modifying them in Step 5!
    const hiddenErrors = [];
    for (let h = 0; h < 4; h++) {
        hiddenErrors.push(outputDelta * brain.weightsHiddenOutput[h][0]);
    }

    // Step 4: Calculate hidden deltas
    // Scale each hidden error by the derivative of that hidden neuron's activation
    const hiddenDeltas = [];
    for (let h = 0; h < 4; h++) {
        hiddenDeltas.push(hiddenErrors[h] * sigmoidDerivative(result.hidden[h]));
    }

    // Step 5: Update hidden -> output weights
    // Weight increases if both the hidden activation and outputDelta are positive
    for (let h = 0; h < 4; h++) {
        brain.weightsHiddenOutput[h][0] += LEARNING_RATE * outputDelta * result.hidden[h];
    }

    // Step 6: Update output bias
    brain.biasOutput[0] += LEARNING_RATE * outputDelta;

    // Step 7: Update input -> hidden weights
    for (let i = 0; i < 2; i++) {
        for (let h = 0; h < 4; h++) {
            brain.weightsInputHidden[i][h] += LEARNING_RATE * hiddenDeltas[h] * input[i];
        }
    }

    // Step 8: Update hidden biases
    for (let h = 0; h < 4; h++) {
        brain.biasHidden[h] += LEARNING_RATE * hiddenDeltas[h];
    }

    // Return squared error for tracking: (target - output)^2
    return Math.pow(target - result.output, 2);
}

// ==================================================
// ERROR EVALUATION
// ==================================================

/**
 * Computes the mean squared error across the ENTIRE current dataset.
 * Does not update any weights; purely evaluates current performance.
 */
function computeAverageError(brain) {
    if (dataset.length === 0) return 0;
    let totalSquaredError = 0;
    for (const example of dataset) {
        const result = forward(brain, example.input);
        totalSquaredError += Math.pow(example.target - result.output, 2);
    }
    return totalSquaredError / dataset.length;
}

// ==================================================
// SELF-TRAINING LOOP
// ==================================================

/**
 * Trains AXEN's brain on the dataset until the average error is below TARGET_ERROR,
 * or until the MAX_TRAINING_STEPS limit is reached.
 */
function train(brain) {
    let step = 0;
    let averageError = computeAverageError(brain);

    // If the brain is already well-trained, report and exit early
    if (averageError < TARGET_ERROR) {
        console.log(`Axen: my average error is ${averageError.toFixed(5)}`);
        return averageError;
    }

    while (step < MAX_TRAINING_STEPS) {
        // 1. Randomly select an example from the dataset
        const randomIndex = Math.floor(Math.random() * dataset.length);
        const example = dataset[randomIndex];

        // 2. Run the forward pass
        const result = forward(brain, example.input);

        // 3, 4, 5. Compare prediction with target and backpropagate weight updates
        backpropagate(brain, example.input, example.target, result);

        step++;

        // Every 1000 training steps, calculate average squared error across entire dataset
        if (step % 1000 === 0) {
            averageError = computeAverageError(brain);
            console.log(`Axen: my average error is ${averageError.toFixed(5)}`);

            // Stop automatically when average error < 0.01
            if (averageError < TARGET_ERROR) {
                break;
            }
        }
    }

    if (step >= MAX_TRAINING_STEPS && averageError >= TARGET_ERROR) {
        console.log("Axen: stopped because the training limit was reached.");
    }

    return averageError;
}

// ==================================================
// BRAIN PERSISTENCE (SAVE / LOAD)
// ==================================================

/**
 * Saves the current weights and biases to axen_brain.json.
 */
function saveBrain(brain) {
    const brainData = {
        weightsInputHidden: brain.weightsInputHidden,
        biasHidden: brain.biasHidden,
        weightsHiddenOutput: brain.weightsHiddenOutput,
        biasOutput: brain.biasOutput
    };
    fs.writeFileSync(BRAIN_FILE, JSON.stringify(brainData, null, 2), "utf8");
}

/**
 * Loads weights and biases from axen_brain.json if it exists.
 * Returns the loaded brain object or null if not found.
 */
function loadBrain() {
    if (fs.existsSync(BRAIN_FILE)) {
        try {
            const raw = fs.readFileSync(BRAIN_FILE, "utf8");
            const data = JSON.parse(raw);
            if (
                Array.isArray(data.weightsInputHidden) &&
                Array.isArray(data.biasHidden) &&
                Array.isArray(data.weightsHiddenOutput) &&
                Array.isArray(data.biasOutput)
            ) {
                return data;
            }
        } catch (err) {
            // Return null if reading or parsing fails
            return null;
        }
    }
    return null;
}

// ==================================================
// PREDICTION & CONFIDENCE
// ==================================================

/**
 * Evaluates an input pair and returns the binary decision and confidence.
 *
 * Confidence calculation:
 * - If prediction >= 0.5: answer is 1, confidence is prediction
 * - If prediction < 0.5: answer is 0, confidence is 1 - prediction
 */
function predict(brain, input) {
    const result = forward(brain, input);
    const prediction = result.output;
    let answer;
    let confidence;

    if (prediction >= 0.5) {
        answer = 1;
        confidence = prediction;
    } else {
        answer = 0;
        confidence = 1 - prediction;
    }

    const percentage = (confidence * 100).toFixed(2);
    return {
        answer: answer,
        confidence: confidence,
        percentage: percentage,
        rawOutput: prediction
    };
}

// ==================================================
// TEACH COMMAND
// ==================================================

/**
 * Teaches AXEN a new (or updated) training example:
 * 1. Adds or updates the example in the dataset.
 * 2. Trains the brain again on the updated dataset.
 * 3. Saves the updated brain.
 * 4. Informs the user in first-person.
 */
function teach(brain, input1, input2, target) {
    // Check if an entry with these exact inputs already exists
    let existingIndex = dataset.findIndex(
        (ex) => ex.input[0] === input1 && ex.input[1] === input2
    );

    if (existingIndex !== -1) {
        dataset[existingIndex].target = target;
    } else {
        dataset.push({ input: [input1, input2], target: target });
    }

    console.log(`Axen: I learned a new example: [${input1}, ${input2}] -> ${target}`);

    // Re-train the brain with the updated dataset
    train(brain);
    console.log("Axen: I trained again.");

    // Save brain state to disk
    saveBrain(brain);
    console.log("Axen: I saved my brain.");
}

// ==================================================
// TERMINAL CHAT INTERFACE
// ==================================================

/**
 * Starts the interactive command-line interface.
 */
function startChat(brain) {
    const rl = readline.createInterface({
        input: process.stdin,
        output: process.stdout,
        prompt: "You: "
    });

    rl.prompt();

    rl.on("line", (line) => {
        const trimmed = line.trim();

        // Ignore empty lines
        if (!trimmed) {
            rl.prompt();
            return;
        }

        // Handle exit command
        if (trimmed.toLowerCase() === "exit") {
            rl.close();
            return;
        }

        // Check for 'teach x1 x2 target'
        const teachMatch = trimmed.match(/^teach\s+(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)$/i);
        if (teachMatch) {
            const in1 = parseFloat(teachMatch[1]);
            const in2 = parseFloat(teachMatch[2]);
            const tgt = parseFloat(teachMatch[3]);
            teach(brain, in1, in2, tgt);
            rl.prompt();
            return;
        }

        // Check for 'x1 x2' prediction query
        const predictMatch = trimmed.match(/^(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)$/);
        if (predictMatch) {
            const in1 = parseFloat(predictMatch[1]);
            const in2 = parseFloat(predictMatch[2]);
            const pred = predict(brain, [in1, in2]);
            console.log(`Axen: I think the answer is ${pred.answer} (${pred.percentage}% sure)`);
            rl.prompt();
            return;
        }

        // Invalid input fallback
        console.log('Axen: I did not understand. Try "1 0" or "teach 1 1 0".');
        rl.prompt();
    });

    rl.on("close", () => {
        process.exit(0);
    });
}

// ==================================================
// MAIN ENTRY POINT
// ==================================================

function main() {
    let brain = loadBrain();

    if (brain) {
        console.log("Axen: I found my old brain. I will continue learning.");
    } else {
        console.log("Axen: I don't have a brain yet. I will create one.");
        brain = createBrain();
    }

    // Train the brain on the XOR dataset
    train(brain);

    // Save the brain after initial training
    saveBrain(brain);

    // Start user interaction
    startChat(brain);
}

// Run main if this file is executed directly
if (require.main === module) {
    main();
}

// Export functions for potential test imports
module.exports = {
    dataset,
    sigmoid,
    sigmoidDerivative,
    randomWeight,
    createBrain,
    forward,
    backpropagate,
    computeAverageError,
    train,
    saveBrain,
    loadBrain,
    predict,
    teach
};
