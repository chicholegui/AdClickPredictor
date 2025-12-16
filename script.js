let model;
let config;

// Load resources on startup
// Main initialization function
// 1. Checks for file protocol restrictions (CORS)
// 2. Loads the TensorFlow.js model
// 3. Fetches the preprocessing configuration JSON
// 4. Initializes UI elements based on config
async function init() {
    // Check for local file execution issue (CORS blocks loading external files like model.json)
    if (window.location.protocol === 'file:') {
        const msg = "⚠️ Security Warning: Browsers block reading files directly (CORS). Please upload to GitHub or use a local server.";
        alert(msg);
        updateStatus(msg);
        // We can't proceed with fetch, so return early or try-catch will handle it
    }

    updateStatus("Loading resources... Step 1: Model");
    try {
        // Load TensorFlow.js Model
        // Added explicit error handling for 404s
        try {
            model = await tf.loadLayersModel('tfjs_model/model.json');
            console.log("Model loaded");
        } catch (modelErr) {
            throw new Error("Failed to load Model. Check 'tfjs_model/model.json' path. Details: " + modelErr.message);
        }

        updateStatus("Loading resources... Step 2: Config");

        // Load Preprocessing Config
        try {
            const response = await fetch('preprocessing_config.json');
            if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
            config = await response.json();
            console.log("Config loaded", config);
        } catch (configErr) {
            throw new Error("Failed to load Config. Check 'preprocessing_config.json'. Details: " + configErr.message);
        }

        // --- POPULATE UI ---
        populateSelect('country', config.label_encoders['Country']);
        populateSelect('adTopic', config.label_encoders['Ad Topic Line']);

        // Dynamic Income Slider Range
        const incomeIdx = config.features_order.indexOf('Area Income');
        if (incomeIdx !== -1) {
            const minInc = config.scaler.data_min[incomeIdx];
            const maxInc = config.scaler.data_max[incomeIdx];
            const incInput = document.getElementById('income');
            incInput.min = Math.floor(minInc);
            incInput.max = Math.ceil(maxInc);
        }

        // --- ATTACH LISTENERS (Auto-Predict) ---
        // Attach 'input' event to all form elements for real-time prediction
        const inputs = document.querySelectorAll('#predictForm input, #predictForm select');
        inputs.forEach(input => {
            input.addEventListener('input', predict);
        });

        // Initial Prediction
        predict();

        updateStatus("Ready - Model is Live");

        if (config.metrics) {
            document.getElementById('accValue').innerText = (config.metrics.accuracy * 100).toFixed(2) + '%';
            document.getElementById('rocValue').innerText = config.metrics.roc_auc.toFixed(4);
        }

    } catch (e) {
        console.error(e);
        const errMsg = "CRITICAL ERROR: " + e.message;
        updateStatus(errMsg);
        alert(errMsg); // Pop-up to ensure user sees it
    }
}

function populateSelect(id, options) {
    const select = document.getElementById(id);
    select.innerHTML = ''; // Clear loading
    // Sort alphabetically for better UX
    options.sort().forEach(opt => {
        const el = document.createElement('option');
        el.value = opt;
        el.innerText = opt;
        select.appendChild(el);
    });
    // Set a random default so it's not always the first one (Afghanistan / ... )
    select.selectedIndex = Math.floor(Math.random() * options.length);
}

function updateStatus(msg) {
    document.getElementById('statusMsg').innerText = msg;
}

// Function is now named 'predict' and called directly on change
async function predict(e) {
    if (e) e.preventDefault(); // Prevent submit if called via event

    // Fail-safe check with visual feedback
    if (!model || !config) {
        // Only alert if this was a manual click, not initial load
        if (e && e.type === 'submit') {
            alert("Model is not ready yet! Please wait for 'Ready' status.");
        } else {
            console.warn("Prediction skipped: Model/Config not loaded.");
        }
        return;
    }

    // updateStatus("Predicting..."); // Optional: too flickery for real-time?

    // Gather inputs
    const dailyTime = parseFloat(document.getElementById('dailyTime').value);
    const age = parseInt(document.getElementById('age').value);
    const income = parseFloat(document.getElementById('income').value);
    const internetUsage = parseFloat(document.getElementById('internetUsage').value);
    const genderStr = document.getElementById('gender').value;
    const countryStr = document.getElementById('country').value;
    const adTopicStr = document.getElementById('adTopic').value;

    // --- DATE FIXED DEFAULT ---
    // The model requires timestamp features (Year, Month, DayOfWeek).
    // Ideally, we would ask the user for a date or use the current date.
    // For consistency in this demo, we use a fixed neutral mid-point (2016-06-01).
    const year = 2016;
    const month = 6;
    const pyDayOfWeek = 2; // Wednesday (0=Mon, ... 2=Wed)

    // 1. Label Encoding
    const encode = (col, val) => {
        const classes = config.label_encoders[col];
        const idx = classes.indexOf(val);
        return idx !== -1 ? idx : 0;
    };

    const countryEncoded = encode('Country', countryStr);
    const topicEncoded = encode('Ad Topic Line', adTopicStr);
    const maleEncoded = genderStr === 'Male' ? 1 : 0;

    // 2. Search Queries Feature Expansion
    // The original model included a list of search queries.
    // We flatten this to match the model's expected input shape.
    // We fill with zeros to simulate a "neutral" search history for stability.
    const numSq = config.num_search_query_features;
    const searchQueries = new Array(numSq).fill(0);

    // 3. Construct Feature Array
    const valueMap = {
        'Daily Time Spent on Site': dailyTime,
        'Age': age,
        'Area Income': income,
        'Daily Internet Usage': internetUsage,
        'Male': maleEncoded,
        'Ad Topic Line': topicEncoded,
        'Country': countryEncoded,
        'Timestamp_Year': year,
        'Timestamp_Month': month,
        'Timestamp_DayOfWeek': pyDayOfWeek
    };

    // Add Query features to map
    for (let i = 0; i < numSq; i++) {
        valueMap[`Search_Query_${i}`] = searchQueries[i];
    }

    const rawFeatures = config.features_order.map(feat => valueMap[feat]);

    // 4. Scaling (MinMax Normalization)
    // We must apply the EXACT same scaling as used during training.
    // (val * scale + min) formula corresponds to sklearn's MinMaxScaler transformation.
    const scaledFeatures = rawFeatures.map((val, idx) => {
        const scale = config.scaler.scale[idx];
        const min = config.scaler.min[idx];
        return val * scale + min;
    });

    // 5. Predict
    // tf.tidy cleans up intermediate tensors automatically
    const prob = tf.tidy(() => {
        const tensor = tf.tensor2d([scaledFeatures]);
        const prediction = model.predict(tensor);
        return prediction.dataSync()[0]; // Sync is fine for small model
    });

    // Update UI
    updateRing(prob);
    // updateStatus(`Probability: ${(prob * 100).toFixed(2)}%`);
}

function updateRing(prob) {
    const ring = document.getElementById('probRing');
    const text = document.getElementById('probPercent');
    const strokeDash = 100;

    const offset = strokeDash - (prob * strokeDash);

    ring.style.strokeDashoffset = offset;
    text.innerText = (prob * 100).toFixed(1) + '%';

    if (prob > 0.5) {
        ring.style.stroke = '#00ff88';
        text.style.color = '#00ff88';
    } else {
        ring.style.stroke = '#ff4444';
        text.style.color = '#ff4444';
    }
}

init();
