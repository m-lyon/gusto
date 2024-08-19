from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme(style='darkgrid')
colours = sns.color_palette()

# Load the CSV file into a DataFrame
RESULTS_DIR = Path(__file__).parent.joinpath('data/unet')
exp1_train_csvs = list(RESULTS_DIR.glob('experiment_1_run*_epoch_loss_train.csv'))
exp1_val_csvs = list(RESULTS_DIR.glob('experiment_1_run*_epoch_loss_validation.csv'))
exp2_20_train_csvs = list(RESULTS_DIR.glob('experiment_2_20pcent_run*_epoch_loss_train.csv'))
exp2_20_val_csvs = list(RESULTS_DIR.glob('experiment_2_20pcent_run*_epoch_loss_validation.csv'))
exp2_40_train_csvs = list(RESULTS_DIR.glob('experiment_2_40pcent_run*_epoch_loss_train.csv'))
exp2_40_val_csvs = list(RESULTS_DIR.glob('experiment_2_40pcent_run*_epoch_loss_validation.csv'))


exp1_train_dfs = [pd.read_csv(csv) for csv in exp1_train_csvs]
for i, df in enumerate(exp1_train_dfs):
    df['experiment'] = i + 1
    df['metric'] = 'train'
exp1_train_df = pd.concat(exp1_train_dfs)
exp2_20_train_dfs = [pd.read_csv(csv) for csv in exp2_20_train_csvs]
for i, df in enumerate(exp2_20_train_dfs):
    df['experiment'] = i + 1
    df['metric'] = 'train'
exp2_20_train_df = pd.concat(exp2_20_train_dfs)
exp2_40_train_dfs = [pd.read_csv(csv) for csv in exp2_40_train_csvs]
for i, df in enumerate(exp2_40_train_dfs):
    df['experiment'] = i + 1
    df['metric'] = 'train'
exp2_40_train_df = pd.concat(exp2_40_train_dfs)

exp1_val_dfs = [pd.read_csv(csv) for csv in exp1_val_csvs]
for i, df in enumerate(exp1_val_dfs):
    df['experiment'] = i + 1
    df['metric'] = 'val'
exp1_val_df = pd.concat(exp1_val_dfs)
exp2_20_val_dfs = [pd.read_csv(csv) for csv in exp2_20_val_csvs]
for i, df in enumerate(exp2_20_val_dfs):
    df['experiment'] = i + 1
    df['metric'] = 'val'
exp2_20_val_df = pd.concat(exp2_20_val_dfs)
exp2_40_val_dfs = [pd.read_csv(csv) for csv in exp2_40_val_csvs]
for i, df in enumerate(exp2_40_val_dfs):
    df['experiment'] = i + 1
    df['metric'] = 'val'
exp2_40_val_df = pd.concat(exp2_40_val_dfs)

min_loss_exp1 = exp1_val_df[exp1_val_df['Value'] == exp1_val_df['Value'].min()][
    'experiment'
].values[0]
best_train_exp1 = exp1_train_df[exp1_train_df['experiment'] == min_loss_exp1]
best_val_exp1 = exp1_val_df[exp1_val_df['experiment'] == min_loss_exp1]

min_loss_exp2_20 = exp2_20_val_df[exp2_20_val_df['Value'] == exp2_20_val_df['Value'].min()][
    'experiment'
].values[0]
best_train_exp2_20 = exp2_20_train_df[exp2_20_train_df['experiment'] == min_loss_exp2_20]
best_val_exp2_20 = exp2_20_val_df[exp2_20_val_df['experiment'] == min_loss_exp2_20]

min_loss_exp2_40 = exp2_40_val_df[exp2_40_val_df['Value'] == exp2_40_val_df['Value'].min()][
    'experiment'
].values[0]
best_train_exp2_40 = exp2_40_train_df[exp2_40_train_df['experiment'] == min_loss_exp2_40]
best_val_exp2_40 = exp2_40_val_df[exp2_40_val_df['experiment'] == min_loss_exp2_40]

# Baseline
baseline_train = pd.read_csv(RESULTS_DIR.joinpath('dict_baseline_step3_train.csv'))
baseline_val = pd.read_csv(RESULTS_DIR.joinpath('dict_baseline_step3_val.csv'))
baseline_train = baseline_train.iloc[::20,]
baseline_val = baseline_val.iloc[::20,]
baseline_train['Step'] = baseline_train['Step'] // 20
baseline_val['Step'] = baseline_val['Step'] // 20

# Create a Seaborn plot
plt.figure(figsize=(10, 6))
sns.lineplot(data=baseline_train, x='Step', y='Value', linestyle='dotted', color=colours[0])
sns.lineplot(
    data=baseline_val,
    x='Step',
    y='Value',
    linestyle='solid',
    color=colours[0],
    label='Dict Learning Baseline',
)
sns.lineplot(data=best_train_exp1, x='Step', y='Value', linestyle='dotted', color=colours[1])
sns.lineplot(
    data=best_val_exp1,
    x='Step',
    y='Value',
    linestyle='solid',
    color=colours[1],
    label='Experiment 1',
)
sns.lineplot(data=best_train_exp2_20, x='Step', y='Value', linestyle='dotted', color=colours[2])
sns.lineplot(
    data=best_val_exp2_20,
    x='Step',
    y='Value',
    linestyle='solid',
    color=colours[2],
    label='Experiment 2 - 20% Pruned',
)
sns.lineplot(data=best_train_exp2_40, x='Step', y='Value', linestyle='dotted', color=colours[3])
sns.lineplot(
    data=best_val_exp2_40,
    x='Step',
    y='Value',
    linestyle='solid',
    color=colours[3],
    label='Experiment 2 - 40% Pruned',
)

plt.xlabel('U-Net Epochs')
plt.ylabel('RMSE')

plt.ylim(0, 0.3)
plt.legend()

ax = plt.gca()
# Create a second x-axis with a different set of labels
ax_top = ax.twiny()  # Create a twin x-axis that shares the y-axis
ax_top.set_xlim(ax.get_xlim())  # Ensure that the top x-axis matches the bottom x-axis

# Set labels for the top x-axis
ax_top.set_xlabel('Dict Learning Epochs')
ax_top.set_xticks([0, 10, 20, 30, 40, 50])
ax_top.set_xticklabels(['0', '200', '400', '600', '800', '1000'])
ax_top.set_zorder(-1)


# Save the plot
plt.savefig(Path(__file__).parent.parent.joinpath('docs', 'unet_experiments_1_to_2.png'))
