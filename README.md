# alphadl - Lightning AI for Cutting-Edge Deep Learning Models in Stock Markets

alphadl leverages Lightning AI to implement and compare various deep learning models for stock market prediction.

## Models

- **StockMixer**: A simple yet powerful MLP-based architecture for stock price forecasting.

## Data Interface

- **OpenBB**: Utilized for accessing comprehensive stock market data.

## Features

- **Efficient and Scalable**: Utilizes Lightning AI for streamlined deep learning workflows.
- **Advanced Prediction Models**: Implements state-of-the-art ML/DL models for stock market forecasting.
- **Data Integration**: Seamlessly integrates OpenBB for data acquisition.
- **Configuration Management**: Employs Hydra for flexible configuration management.
- **Experiment Tracking**: Integrates Weights & Biases (wandb) for tracking and visualizing experiments.

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/alphadl.git
   cd alphadl
   ```

2. **Install Required Packages**:
   ```bash
   pip install -e .
   ```

## How to Execute

Run the training script with a specified experiment configuration:

```bash
python src/train experiment=<experiment_name>
```

For example:

```bash
python src/train experiment=stockmixer_sp500_1d
```

## Contributions

We welcome contributions to alphadl! Here's how you can get involved:

1. **Fork the Repository**: Create a new branch for your feature or bug fix.
2. **Code Style**: Follow existing code style and formatting conventions.
3. **Testing**: Write unit tests to ensure your changes work as expected.
4. **Submit a Pull Request**: Provide a clear description of your changes.

We appreciate all contributions, whether it's a bug fix, new feature, or documentation improvement. If you're unsure where to start, reach out on our issue tracker or discussion forum.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions or feedback, please contact us at [igorlima1740@gmail.com](mailto:igorlima1740@gmail.com).


