import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO
from sklearn.datasets import load_iris


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def train_gradient_descent(X, Y, w1, w2, w3, w4, w5, w6, w7, w8, bh1, bh2, bo, lr, epochs):
    """Train the neural network with gradient descent"""
    error_history = []
    output_history = []
    
    for epoch in range(epochs):
        total_error = 0
        for i in range(len(X)):
            x1, x2, x3 = X[i][0], X[i][1], X[i][2]
            
            # Forward pass
            zh1 = w1*x1 + w3*x2 + w5*x3 + bh1
            zh2 = w2*x1 + w4*x2 + w6*x3 + bh2
            
            h1 = sigmoid(zh1)
            h2 = sigmoid(zh2)
            
            z0 = w7*h1 + w8*h2 + bo
            o = sigmoid(z0)
            
            # Error calculation
            error = (Y[i] - o)**2
            total_error += error
            
            # Backpropagation
            delta_o = error * o * (1 - o)
            delta_h1 = delta_o * w7 * h1 * (1 - h1)
            delta_h2 = delta_o * w8 * h2 * (1 - h2)
            
            # Update weights and biases
            w7 += lr * delta_o * h1
            w8 += lr * delta_o * h2
            bo += lr * delta_o
            w1 += lr * delta_h1 * x1
            w3 += lr * delta_h1 * x2
            w5 += lr * delta_h1 * x3
            bh1 += lr * delta_h1
            w2 += lr * delta_h2 * x1
            w4 += lr * delta_h2 * x2
            w6 += lr * delta_h2 * x3
            bh2 += lr * delta_h2
            
        error_history.append(total_error)
        output_history.append(o)
    
    weights = {
        'w1': w1, 'w2': w2, 'w3': w3, 'w4': w4, 'w5': w5, 'w6': w6,
        'w7': w7, 'w8': w8, 'bh1': bh1, 'bh2': bh2, 'bo': bo
    }
    
    return error_history, output_history, weights


def main():
    st.set_page_config(page_title="Gradient Descent Neural Network Simulator", layout="wide")
    
    st.markdown("<h1 style='text-align: center; font-size: 28px; margin-bottom: 5px;'>🧠 GD Neural Network</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size: 11px; margin-top: 0; margin-bottom: 15px; color: #888;'>Real-time Training Visualization</p>", unsafe_allow_html=True)
    
    # Create three main columns
    col_data, col_weights, col_plot = st.columns([1, 1, 1.2])
    
    # ========== DATA COLUMN ==========
    with col_data:
        st.markdown("<h3 style='font-size: 16px; margin-bottom: 10px;'>📁 Data</h3>", unsafe_allow_html=True)
        
        # Data source selector
        data_source = st.radio(
            "Select Data Source",
            ["📄 CSV Upload", "🌸 Iris Dataset", "📦 Default"],
            horizontal=True,
            label_visibility="collapsed"
        )
        
        if data_source == "📄 CSV Upload":
            # File uploader
            uploaded_file = st.file_uploader("Upload CSV file", type=['csv'], label_visibility="collapsed")
            
            if uploaded_file is not None:
                df = pd.read_csv(uploaded_file)
                st.success(f"✅ {uploaded_file.name}")
                
                # Extract X and Y
                if df.shape[1] >= 4:
                    X = df.iloc[:, :3].values.tolist()
                    Y = df.iloc[:, 3].values.tolist()
                    st.info(f"📊 {df.shape[0]} samples loaded")
                else:
                    st.error("CSV must have at least 4 columns (3 features + 1 target)")
                    X, Y = None, None
            else:
                st.warning("⬆️ Please upload a CSV file")
                X, Y = None, None
                
        elif data_source == "🌸 Iris Dataset":
            # Load Iris dataset
            iris = load_iris()
            # Use first 3 features (sepal length, sepal width, petal length)
            X_iris = iris.data[:, :3]
            # Binary classification: setosa (0) vs non-setosa (1)
            Y_iris = (iris.target != 0).astype(int)
            
            X = X_iris.tolist()
            Y = Y_iris.tolist()
            
            st.success(f"✅ Iris Dataset Loaded")
            st.info(f"📊 {len(X)} samples | Binary: Setosa (0) vs Others (1)")
            
            # Show sample data
            with st.expander("👁️ Preview Data", expanded=False):
                preview_df = pd.DataFrame(X_iris[:10], columns=['Sepal Length', 'Sepal Width', 'Petal Length'])
                preview_df['Target'] = Y_iris[:10]
                st.dataframe(preview_df, use_container_width=True, hide_index=True)
        
        else:  # Default
            st.info("📦 Default dataset active")
            X = [[2,60,45], [3,65,50], [4,70,55], [5,75,60], 
                 [6,80,65], [7,85,70], [8,90,75]]
            Y = [0,0,0,1,1,1,1]
        
        st.markdown("<h3 style='font-size: 16px; margin-bottom: 10px; margin-top: 10px;'>⚙️ Hyperparameters</h3>", unsafe_allow_html=True)
        
        lr = st.slider("🎯 Learning Rate", 0.001, 0.5, 0.05, 0.001)
        epochs = st.slider("🔄 Epochs", 50, 2000, 500, 50)
        
        # Training button
        train_button = st.button("🚀 TRAIN MODEL", type="primary", use_container_width=True)
    
    # ========== WEIGHTS COLUMN ==========
    with col_weights:
        st.markdown("<h3 style='font-size: 16px; margin-bottom: 10px;'>🔧 Weights</h3>", unsafe_allow_html=True)
        
        # Weight initialization selector
        weight_mode = st.radio(
            "Weight Initialization",
            ["🎲 Random", "🎛️ Manual"],
            horizontal=True,
            label_visibility="collapsed"
        )
        
        if weight_mode == "🎲 Random":
            # Random weight initialization
            st.info("🎲 Weights initialized randomly")
            np.random.seed(42)  # For reproducibility
            w1, w2 = np.random.randn(2) * 0.5
            w3, w4 = np.random.randn(2) * 0.5
            w5, w6 = np.random.randn(2) * 0.5
            w7, w8 = np.random.randn(2) * 0.5
            bh1, bh2 = np.random.randn(2) * 0.5
            bo = np.random.randn() * 0.5
            
            # Show random values
            with st.expander("👁️ View Random Weights", expanded=False):
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Input → Hidden**")
                    st.code(f"w1={w1:.3f}, w2={w2:.3f}\nw3={w3:.3f}, w4={w4:.3f}\nw5={w5:.3f}, w6={w6:.3f}\nbh1={bh1:.3f}, bh2={bh2:.3f}")
                with col2:
                    st.write("**Hidden → Output**")
                    st.code(f"w7={w7:.3f}, w8={w8:.3f}\nbo={bo:.3f}")
        
        else:  # Manual weights
            with st.expander("⚡ Input → Hidden Layer Weights", expanded=False):
                col1, col2, col3 = st.columns(3)
                with col1:
                    w1 = st.slider("w1", -1.0, 1.0, 0.1, 0.1, key="w1")
                    w2 = st.slider("w2", -1.0, 1.0, -0.2, 0.1, key="w2")
                with col2:
                    w3 = st.slider("w3", -1.0, 1.0, 0.4, 0.1, key="w3")
                    w4 = st.slider("w4", -1.0, 1.0, 0.2, 0.1, key="w4")
                with col3:
                    w5 = st.slider("w5", -1.0, 1.0, -0.5, 0.1, key="w5")
                    w6 = st.slider("w6", -1.0, 1.0, 0.1, 0.1, key="w6")
                
                col1, col2 = st.columns(2)
                with col1:
                    bh1 = st.slider("bias h1", -1.0, 1.0, 0.1, 0.1, key="bh1")
                with col2:
                    bh2 = st.slider("bias h2", -1.0, 1.0, -0.1, 0.1, key="bh2")
            
            with st.expander("🎯 Hidden → Output Layer Weights", expanded=False):
                col1, col2, col3 = st.columns(3)
                with col1:
                    w7 = st.slider("w7", -1.0, 1.0, 0.3, 0.1, key="w7")
                with col2:
                    w8 = st.slider("w8", -1.0, 1.0, -0.3, 0.1, key="w8")
                with col3:
                    bo = st.slider("bias output", -1.0, 1.0, 0.2, 0.1, key="bo")
    
    # ========== PLOT COLUMN ==========
    with col_plot:
        st.markdown("<h3 style='font-size: 16px; margin-bottom: 10px;'>📊 Results</h3>", unsafe_allow_html=True)
        
        if train_button and X is not None and Y is not None:
            with st.spinner('🔄 Training in progress...'):
                error_history, output_history, final_weights = train_gradient_descent(
                    X, Y, w1, w2, w3, w4, w5, w6, w7, w8, bh1, bh2, bo, lr, epochs
                )
            
            st.success(f"✅ Training Completed | Error: {error_history[-1]:.6f}")
            
            # Combined Chart with dual y-axes
            fig, ax1 = plt.subplots(figsize=(5, 4))
            
            # Error line (left y-axis)
            color1 = '#FF6B6B'
            ax1.set_xlabel('Epoch', fontsize=9, fontweight='bold')
            ax1.set_ylabel('Total Error', color=color1, fontsize=9, fontweight='bold')
            line1 = ax1.plot(error_history, linewidth=2, color=color1, label='Error', alpha=0.9)
            ax1.tick_params(axis='y', labelcolor=color1, labelsize=8)
            ax1.tick_params(axis='x', labelsize=8)
            ax1.grid(True, alpha=0.2, linestyle='--')
            ax1.set_facecolor('#f8f9fa')
            
            # Output line (right y-axis)
            ax2 = ax1.twinx()
            color2 = '#4ECDC4'
            ax2.set_ylabel('Output', color=color2, fontsize=9, fontweight='bold')
            line2 = ax2.plot(output_history, linewidth=2, color=color2, label='Output', alpha=0.9)
            ax2.tick_params(axis='y', labelcolor=color2, labelsize=8)
            
            # Title and legend
            fig.suptitle('Training Convergence', fontsize=11, fontweight='bold')
            
            # Combine legends
            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            ax1.legend(lines, labels, loc='upper right', framealpha=0.95, fontsize=8)
            
            fig.tight_layout()
            st.pyplot(fig)
            
            # Metrics row (compact text)
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.markdown(f"<div style='text-align:center; font-size: 10px; color:#999;'>Error</div><div style='text-align:center; font-size: 12px; font-weight:600;'>{error_history[-1]:.4f}</div>", unsafe_allow_html=True)
            with col2:
                st.markdown(f"<div style='text-align:center; font-size: 10px; color:#999;'>Output</div><div style='text-align:center; font-size: 12px; font-weight:600;'>{output_history[-1]:.4f}</div>", unsafe_allow_html=True)
            with col3:
                st.markdown(f"<div style='text-align:center; font-size: 10px; color:#999;'>Epochs</div><div style='text-align:center; font-size: 12px; font-weight:600;'>{epochs}</div>", unsafe_allow_html=True)
            with col4:
                st.markdown(f"<div style='text-align:center; font-size: 10px; color:#999;'>LR</div><div style='text-align:center; font-size: 12px; font-weight:600;'>{lr}</div>", unsafe_allow_html=True)
            
            # Display final weights in compact format
            with st.expander("🏁 Final Weights", expanded=False):
                col1, col2 = st.columns(2)
                with col1:
                    st.write("<p style='font-size: 11px; margin-bottom: 5px;'><b>Input → Hidden</b></p>", unsafe_allow_html=True)
                    weights_ih = pd.DataFrame({
                        'Weight': ['w1', 'w2', 'w3', 'w4', 'w5', 'w6', 'bh1', 'bh2'],
                        'Value': [f"{final_weights['w1']:.4f}", f"{final_weights['w2']:.4f}", 
                                 f"{final_weights['w3']:.4f}", f"{final_weights['w4']:.4f}", 
                                 f"{final_weights['w5']:.4f}", f"{final_weights['w6']:.4f}",
                                 f"{final_weights['bh1']:.4f}", f"{final_weights['bh2']:.4f}"]
                    })
                    st.dataframe(weights_ih, use_container_width=True, hide_index=True)
                
                with col2:
                    st.write("<p style='font-size: 11px; margin-bottom: 5px;'><b>Hidden → Output</b></p>", unsafe_allow_html=True)
                    weights_ho = pd.DataFrame({
                        'Weight': ['w7', 'w8', 'bo'],
                        'Value': [f"{final_weights['w7']:.4f}", f"{final_weights['w8']:.4f}", 
                                 f"{final_weights['bo']:.4f}"]
                    })
                    st.dataframe(weights_ho, use_container_width=True, hide_index=True)
        else:
            st.info("👈 Configure data & weights, then click TRAIN MODEL")


if __name__ == "__main__":
    main()
