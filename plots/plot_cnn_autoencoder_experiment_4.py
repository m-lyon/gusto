from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme(style='darkgrid')

# Load the CSV file into a DataFrame
RESULTS_DIR = Path(__file__).parent.joinpath('data/cnn_autoencoder')
exp1_train = pd.read_csv(RESULTS_DIR.joinpath('experiment_1_opt_epoch_loss_train.csv'))
exp1_val = pd.read_csv(RESULTS_DIR.joinpath('experiment_1_opt_epoch_loss_validation.csv'))
exp2_train = pd.read_csv(RESULTS_DIR.joinpath('experiment_2_opt_epoch_loss_train.csv'))
exp2_val = pd.read_csv(RESULTS_DIR.joinpath('experiment_2_opt_epoch_loss_validation.csv'))
exp3_train = pd.read_csv(RESULTS_DIR.joinpath('experiment_3_opt_epoch_loss_train.csv'))
exp3_val = pd.read_csv(RESULTS_DIR.joinpath('experiment_3_opt_epoch_loss_validation.csv'))

colours = sns.color_palette()

# Create a Seaborn plot
plt.figure(figsize=(10, 6))
sns.lineplot(data=exp1_train.iloc[::5,], x='Step', y='Value', linestyle='dotted', color=colours[0])
sns.lineplot(
    data=exp1_val.iloc[::5,],
    x='Step',
    y='Value',
    linestyle='solid',
    color=colours[0],
    label='Experiment 1',
)
sns.lineplot(data=exp2_train.iloc[::5,], x='Step', y='Value', linestyle='dotted', color=colours[1])
sns.lineplot(
    data=exp2_val.iloc[::5,],
    x='Step',
    y='Value',
    linestyle='solid',
    color=colours[1],
    label='Experiment 2',
)
sns.lineplot(data=exp3_train.iloc[::5,], x='Step', y='Value', linestyle='dotted', color=colours[2])
sns.lineplot(
    data=exp3_val.iloc[::5,],
    x='Step',
    y='Value',
    linestyle='solid',
    color=colours[2],
    label='Experiment 3',
)

# Add title and labels
# plt.title('Step vs Value')
plt.xlabel('Epochs')
plt.ylabel('RMSE')
plt.legend()

# Save the plot
plt.savefig('plot_experiments_4.png')
