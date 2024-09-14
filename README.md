# 🎮 Cart-Pole Control using Deep Q-Network (DQN)

This project implements a reinforcement learning agent to solve the classic **Cart-Pole** problem using a **Deep Q-Network (DQN)**. The project uses the `gym` environment from OpenAI to simulate the Cart-Pole game and trains a neural network to balance the pole by taking appropriate actions.

## 📚 Overview

![AI Cart-Pole Balancing](cartpole_balance.png)


The Cart-Pole problem is a well-known reinforcement learning challenge where the goal is to keep a pole balanced on a moving cart. The agent learns to balance the pole using a deep learning-based Q-network. 

### **Reinforcement Learning Basics**
- **Agent**: Interacts with the environment and makes decisions based on its current state.
- **Environment**: The Cart-Pole simulation where the agent learns and receives feedback.
- **Rewards**: The agent receives positive rewards for keeping the pole balanced and penalties when the pole falls.

## 🛠 Project Components

### 1. **Environment Setup**
- Utilizes OpenAI's `gym` to create the Cart-Pole environment (`CartPole-v1`).
- The environment provides:
  - **State**: A set of four values representing the cart's position, velocity, pole angle, and angular velocity.
  - **Actions**: Two discrete actions - moving the cart left or right.

### 2. **Neural Network Architecture**
- The Q-Network is built using TensorFlow and consists of three hidden layers with ReLU activation functions.
- The network's output layer provides the Q-values for each possible action, which the agent uses to select the best action based on the current state.

### 3. **Key Components in Reinforcement Learning**
- **Policy**: The agent’s strategy for choosing actions based on the current state.
- **Reward**: The immediate feedback from the environment used to evaluate actions.
- **Value Function**: An estimate of the long-term reward for each state.
- **Action-Value (Q-Value)**: The expected return of a specific action taken in a particular state.
- **Markov Decision Process (MDP)**: Used to model the decision-making process for the agent in the environment.

### 4. **Hyperparameters**
- **Training Episodes**: 700
- **Max Steps per Episode**: 500
- **Discount Factor (Gamma)**: 1.0
- **Exploration**: 
  - Start Probability: 1.0 (fully exploring)
  - Minimum Probability: 0.01
  - Decay Rate: 0.0002
- **Network Structure**: Hidden layers with 64 units each.
- **Learning Rate**: 0.0001
- **Memory Buffer Size**: 100,000 for experience replay.
- **Batch Size**: 32

### 5. **Training Process**
- Utilizes **Q-learning** with experience replay to break the correlation between consecutive actions.
- Stores past experiences (state, action, reward, next state) in a memory buffer and randomly samples batches for training.
- Uses the Bellman equation to update Q-values and optimize the network.

### 6. **Visualization of Training Progress**
- Plots total rewards per episode to show the agent's learning progress. As training proceeds, the rewards increase, indicating improved performance in balancing the pole.

### 7. **Testing the Trained Agent**
- After training, the model is tested by running simulations in the environment to visualize the agent's performance using the learned policy.

## 📦 Requirements
- Python 3.x
- `gym`
- `numpy`
- `tensorflow`
- `matplotlib`

## 🚀 Getting Started
1. **Clone the repository**:
    ```bash
    git clone https://github.com/YourUsername/CartPole-DQN.git
    cd CartPole-DQN
    ```

2. **Install the required packages**:
    ```bash
    pip install -r requirements.txt
    ```

3. **Run the training**:
    - Open the Jupyter Notebook and run the cells to start training the agent.
    - Modify hyperparameters as needed to experiment with different configurations.

4. **Visualize Training Progress**:
    - The training process includes a plot of the total rewards per episode to monitor the agent's learning.

5. **Watch the Trained Agent**:
    - Run the `watch_agent` function to see the trained agent in action.

## 📈 Results
- The agent successfully learns to balance the pole for increasingly longer periods.
- The training results show a steady increase in rewards as the agent refines its decision-making strategy.

## 📊 Methodology and Libraries Used
- **TensorFlow**: Used to build and train the neural network that approximates the Q-function for the Cart-Pole problem.
- **NumPy**: For numerical computations and state representation.
- **OpenAI Gym**: Provides the Cart-Pole simulation environment.
- **Matplotlib**: For visualizing training progress and plotting the total rewards.

## 📝 Notes
- **Experience Replay**: Implemented using a circular buffer to store past experiences, which improves the stability and efficiency of training by randomly sampling past events.
- **Model Saving**: The trained model is saved and can be reloaded for further use or evaluation.

## 🔗 References
- OpenAI Gym: [https://gym.openai.com/](https://gym.openai.com/)
- TensorFlow: [https://www.tensorflow.org/](https://www.tensorflow.org/)
- [Reinforcement Learning: An Introduction](https://www.coursera.org/lecture/fundamentals-of-reinforcement-learning/)

## ⚖️ Evaluation
- The project's results show that reinforcement learning using DQNs can effectively balance the Cart-Pole.
- Future work may involve applying these techniques to more complex environments to further enhance the agent's learning capabilities.

## 💬 Let's Connect
- **LinkedIn**: [Shubham Satish Kumar Mishra](https://www.linkedin.com/in/shubham-mishra-025423246/)
- **Email**: [shubh07062002@gmail.com](mailto:shubh07062002@gmail.com)
- **Padlet**: [Shubham's Padlet Profile](https://padlet.com/w9641416/shubham-s-padlet-pitz7jqci9smj4fi)

Feel free to explore, modify, and experiment with the code to enhance the agent's performance! If you have any questions, feel free to reach out.

---

### 👨‍💻 Author
Shubham Satish Kumar Mishra
