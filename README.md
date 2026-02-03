# 🧠 Gradient Descent Neural Network Simulator

An interactive Streamlit dashboard for training and visualizing a simple neural network using gradient descent.

## 🎯 Features

### Left Panel - Data Input
- **CSV Upload**: Upload your own dataset (3 features + 1 target column)
- **Schema Validation**: Automatic detection of data shape and columns
- **Data Preview**: View your uploaded data in a table
- **Default Dataset**: Uses a built-in dataset if no file is uploaded

### Right Panel - Model Parameters
- **Hyperparameters**: 
  - Learning Rate (0.001 - 0.5)
  - Epochs (50 - 2000)
  
- **Weights & Biases** (all adjustable via sliders):
  - Input → Hidden Layer: w1-w6, bias h1, bias h2
  - Hidden → Output Layer: w7, w8, bias output

### Live Visualization
- **Error Convergence Plot**: Real-time training error over epochs
- **Output Evolution Plot**: Model output progression during training
- **Final Weights Display**: Table showing all trained weights
- **Predictions Table**: Compare predictions vs actual labels
- **Accuracy Metric**: Model performance on training data

## 🏗️ Architecture

- **Input Layer**: 3 neurons (3 features)
- **Hidden Layer**: 2 neurons with sigmoid activation
- **Output Layer**: 1 neuron with sigmoid activation (binary classification)
- **Training**: Custom gradient descent with backpropagation

## 📁 Structure

```
.
├── app.py              # Streamlit dashboard (main application)
├── gd_backend.py       # Original gradient descent implementation
└── README.md           # This file
```

## 🚀 Installation & Run

1. **Install dependencies**:
```bash
pip install streamlit numpy pandas matplotlib
```

2. **Run the application**:
```bash
streamlit run app.py
```

3. **Open browser**: The app will automatically open at `http://localhost:8501`

## 📊 CSV Format

Your CSV file should have:
- **3 feature columns** (any numerical values)
- **1 target column** (binary: 0 or 1)

Example:
```csv
feature1,feature2,feature3,target
2,60,45,0
3,65,50,0
4,70,55,0
5,75,60,1
6,80,65,1
7,85,70,1
8,90,75,1
```

## 🎮 Usage

1. Upload a CSV file or use the default dataset
2. Adjust model parameters using the sliders
3. Click "🚀 Train Model" to start training
4. Watch the live plots update
5. Analyze the results: error convergence, predictions, and accuracy

## 🔧 Technical Details

- **Activation Function**: Sigmoid (logistic)
- **Loss Function**: Mean Squared Error (MSE)
- **Optimization**: Stochastic Gradient Descent (SGD)
- **Backpropagation**: Manual implementation for educational purposes

## 💡 Tips

- Start with default parameters to see baseline performance
- Lower learning rates (0.01-0.05) provide stable convergence
- Higher epochs allow more training iterations
- Adjust initial weights to see how they affect convergence
- Try different datasets to test generalization

## 📝 Notes

This is an educational project demonstrating:
- Neural network fundamentals
- Gradient descent optimization
- Backpropagation algorithm
- Interactive ML visualization
- Real-time parameter tuning

---

**Built with**: Streamlit, NumPy, Pandas, Matplotlib
