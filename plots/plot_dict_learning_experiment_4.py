from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme(style='darkgrid')

# Load the CSV file into a DataFrame

RESULTS_DIR = Path(__file__).parent.joinpath('data/dict_learning')
exp4_train_32 = pd.read_csv(RESULTS_DIR.joinpath('experiment_4_step3_32_train.csv'))
exp4_val_32 = pd.read_csv(RESULTS_DIR.joinpath('experiment_4_step3_32_validation.csv'))
exp4_train_64 = pd.read_csv(RESULTS_DIR.joinpath('experiment_4_step3_64_train.csv'))
exp4_val_64 = pd.read_csv(RESULTS_DIR.joinpath('experiment_4_step3_64_validation.csv'))
exp4_train_128 = pd.read_csv(RESULTS_DIR.joinpath('experiment_4_step3_128_train.csv'))
exp4_val_128 = pd.read_csv(RESULTS_DIR.joinpath('experiment_4_step3_128_validation.csv'))
exp4_train_256 = pd.read_csv(RESULTS_DIR.joinpath('experiment_4_step3_256_train.csv'))
exp4_val_256 = pd.read_csv(RESULTS_DIR.joinpath('experiment_4_step3_256_validation.csv'))

colours = sns.color_palette()

# Create a Seaborn plot
plt.figure(figsize=(10, 6))
sns.lineplot(
    data=exp4_train_32.iloc[::5,], x='Step', y='Value', linestyle='dotted', color=colours[0]
)
sns.lineplot(
    data=exp4_val_32.iloc[::5,],
    x='Step',
    y='Value',
    linestyle='solid',
    color=colours[0],
    label='32',
)
sns.lineplot(
    data=exp4_train_64.iloc[::5,], x='Step', y='Value', linestyle='dotted', color=colours[1]
)
sns.lineplot(
    data=exp4_val_64.iloc[::5,],
    x='Step',
    y='Value',
    linestyle='solid',
    color=colours[1],
    label='64',
)
sns.lineplot(
    data=exp4_train_128.iloc[::5,], x='Step', y='Value', linestyle='dotted', color=colours[2]
)
sns.lineplot(
    data=exp4_val_128.iloc[::5,],
    x='Step',
    y='Value',
    linestyle='solid',
    color=colours[2],
    label='128',
)
sns.lineplot(
    data=exp4_train_256.iloc[::5,], x='Step', y='Value', linestyle='dotted', color=colours[3]
)
sns.lineplot(
    data=exp4_val_256.iloc[::5,],
    x='Step',
    y='Value',
    linestyle='solid',
    color=colours[3],
    label='256',
)

# Add title and labels
# plt.title('Step vs Value')
plt.xlabel('Epochs')
plt.ylabel('RMSE')
plt.legend()
plt.ylim(0, 0.30)

# Save the plot
plt.savefig(Path(__file__).parent.parent.joinpath('docs', 'dict_learning_experiments_4.png'))
