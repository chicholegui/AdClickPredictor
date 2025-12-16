# AdClick Neural Visualizer (Beta) 🧠📊

> **Live Demo:** [Check out the App](https://chicholegui.github.io/AdClickPredictor/)

## 🚀 Overview

This project is an experimental web interface developed for personal curiosity, after training a Neural network for a **Deep Learning University Exam**. 

The goal was to take a standard Neural Network model (trained for my DL exam based on the *Ad Click Prediction* dataset) and try to make it interactable user interface, rather than just leaving it in a static Python notebook. 

It is a **Beta** version designed to visualize how the model makes decisions in real-time.

## 🎯 The Idea

I wanted to explore how we can peek inside the "black box" of a trained model:
- **Interactive Visualization:** Experimenting with how changing inputs (like Age or Time on Site) affects the model's confidence in real-time.
- **Model Behavior:** You might notice some extreme predictions (e.g., close to 100% probability for very low internet usage, less % for younger ages and viceversa). This reflects what the model learned from the exam dataset, including its biases and outliers.

## 🛠️ Tools Used

I used this project to practice connecting Python training with Web deployment:

*   **Python & Keras:** For training the initial neural network and preprocessing the data (Pandas/Scikit-Learn).
*   **TensorFlow.js:** To run the saved model directly in the browser.
*   **Web Technologies:** Simple HTML/CSS/JS to build the interface and handle the logic without a backend.

---
*Created by Gabriel Leguizamon.*
