from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme(style='darkgrid')
colours = sns.color_palette()

# Load the CSV file into a DataFrame
RESULTS_DIR = Path(__file__).parent.joinpath('data/unet')
exp3_train_csvs = list(RESULTS_DIR.glob('experiment_3_run*_epoch_loss_train.csv'))
exp3_val_csvs = list(RESULTS_DIR.glob('experiment_3_run*_epoch_loss_validation.csv'))
exp3_val_var_csvs = list(RESULTS_DIR.glob('experiment_3_run*_epoch_variance_validation.csv'))


exp3_train_dfs = [pd.read_csv(csv) for csv in exp3_train_csvs]
for i, df in enumerate(exp3_train_dfs):
    df['experiment'] = i + 1
    df['metric'] = 'train'
exp3_train_df = pd.concat(exp3_train_dfs)
exp3_train_df = exp3_train_df[exp3_train_df['Step'] < 101]

exp3_val_dfs = [pd.read_csv(csv) for csv in exp3_val_csvs]
for i, df in enumerate(exp3_val_dfs):
    df['experiment'] = i + 1
    df['metric'] = 'val'
exp3_val_df = pd.concat(exp3_val_dfs)
exp3_val_df = exp3_val_df[exp3_val_df['Step'] < 101]

exp3_val_var_dfs = [pd.read_csv(csv) for csv in exp3_val_var_csvs]
for i, df in enumerate(exp3_val_var_dfs):
    df['experiment'] = i + 1
    df['metric'] = 'val'
exp3_val_var_df = pd.concat(exp3_val_var_dfs)
exp3_val_var_df = exp3_val_var_df[exp3_val_var_df['Step'] < 101]


min_loss_exp3 = exp3_val_df[exp3_val_df['Value'] == exp3_val_df['Value'].min()][
    'experiment'
].values[0]
best_train_exp3 = exp3_train_df[exp3_train_df['experiment'] == min_loss_exp3]
best_val_exp3 = exp3_val_df[exp3_val_df['experiment'] == min_loss_exp3]
best_val_var_exp3 = exp3_val_var_df[exp3_val_var_df['experiment'] == min_loss_exp3]

# Create a Seaborn plot
plt.figure(figsize=(10, 6))
sns.lineplot(
    data=best_train_exp3,
    x='Step',
    y='Value',
    linestyle='dotted',
    color=colours[0],
    label='Training RMSE',
)
sns.lineplot(
    data=best_val_exp3,
    x='Step',
    y='Value',
    linestyle='solid',
    color=colours[0],
    label='Validation RMSE',
)
sns.lineplot(
    data=best_val_var_exp3,
    x='Step',
    y='Value',
    linestyle='dashdot',
    color=colours[0],
    label=r'Validation $\sigma$',
)

plt.xlabel('Epochs')
plt.ylabel('')

plt.ylim(0, 0.3)
plt.legend()

ax = plt.gca()

# Save the plot
plt.savefig(Path(__file__).parent.parent.joinpath('docs', 'unet_experiments_3.png'))
