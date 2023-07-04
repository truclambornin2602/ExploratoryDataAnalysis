def unistats(df):
    import pandas as pd
    output_df = pd.DataFrame(columns=['Count', 'Missing', 'Unique', 'Dtype', 'Numeric', 'Mode', 'Mean',  'Min', '25%', 'Median', '75%', 'Max', 'Std', 'Skew', 'Kurt'])

    for col in df:
        if pd.api.types.is_numeric_dtype(df[col]) :
            output_df.loc[col] = [df[col].count(), df[col].isnull().sum(), df[col].nunique(), df[col].dtype, pd.api.types.is_numeric_dtype(df[col]),
                                  df[col].mode().values[0], df[col].mean(), df[col].min(), df[col].quantile(0.25), df[col].median(), df[col].quantile(0.75),
                                  df[col].max(), df[col].std(), df[col].skew(), df[col].kurt()]
        else:
            output_df.loc[col] = [df[col].count(), df[col].isnull().sum(), df[col].nunique(), df[col].dtype, pd.api.types.is_numeric_dtype(df[col]),
                                  df[col].mode().values[0], '-', '-', '-', '-', '-', '-', '-', '-', '-']

    return output_df.sort_values(by=['Numeric', 'Skew', 'Unique'], ascending=False)


def anova(df, feature, label):
  import pandas as pd
  import numpy as np
  from scipy import stats

  groups = df[feature].unique()
  df_grouped = df.groupby(feature)
  group_labels = []
  for g in groups:
    g_list = df_grouped.get_group(g)
    group_labels.append(g_list[label])

  return stats.f_oneway(*group_labels)


def heteroscedasticity(df, feature, label):
  from statsmodels.stats.diagnostic import het_breuschpagan
  from statsmodels.stats.diagnostic import het_white
  import pandas as pd
  import statsmodels.api as sm
  from statsmodels.formula.api import ols

  model = ols(formula=(label + '~' + feature), data=df).fit()

  white_test = het_white(model.resid, model.model.exog)
  bp_test = het_breuschpagan(model.resid, model.model.exog)

  output_df = pd.DataFrame(columns=['LM stat', 'LM p-value', 'F-stat', 'F p-value'])
  output_df.loc['White'] = white_test
  output_df.loc['Breusch-Pagan'] = bp_test
  return output_df.round(3)


def scatter(feature, label):
  import seaborn as sns
  from scipy import stats
  import matplotlib.pyplot as plt
  import pandas as pd

  # Calculate the regression line
  m, b, r, p , err = stats.linregress(feature, label)

  textstr = 'y = ' + str(round(m, 2)) + 'x + ' + str(round(b, 2)) + '\n'
  textstr += 'r2 = ' + str(round(r**2, 2)) + '\n'
  textstr += 'p = ' + str(round(p, 2)) + '\n'
  textstr += str(feature.name) + ' skew = ' + str(round(feature.skew(), 2)) + '\n'
  textstr += str(label.name) + ' skew = ' + str(round(feature.skew(), 2)) + '\n'
  textstr += str(heteroscedasticity(pd.DataFrame(label).join(pd.DataFrame(feature)), feature.name, label.name))

  df = pd.DataFrame({'Feature': feature, 'Label': label})
  sns.set(color_codes=True)
  ax = sns.jointplot(data=df, x= feature, y= label, kind='reg')
  ax.fig.text(1, 0.13, textstr, fontsize=12, transform=plt.gcf().transFigure)


def bivstats(df, label):
  from scipy import stats
  import pandas as pd
  import numpy as np

  # Create an empty Dataframe to store the output
  output_df = pd.DataFrame(columns=['Stat', '+/-', 'Effect size', 'p-values'])

  for col in df:
    if not col == label:
      if df[col].isnull().sum() == 0:
        if pd.api.types.is_numeric_dtype(df[col]):
          r, p = stats.pearsonr(df[label], df[col])
          output_df.loc[col] = ['r', str(np.sign(r))[:-6], abs(round(r, 3)), round(p, 6)]
          scatter(df[col], df[label])
        else:
          F, p = anova(df[[col, label]], col, label)
          output_df.loc[col] = ['F', '', round(F, 3), round(p, 6)]
      else:
        output_df.loc[col] = [np.nan, np.nan, np.nan, np.nan]

  return output_df.sort_values(by=['Effect size', 'Stat'], ascending=[False, False])