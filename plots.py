import plotly.express as px
import pandas as pd

pd.pandas.set_option('display.max_columns', None)
pd.set_option('expand_frame_repr', False)
pd.options.display.expand_frame_repr = False
pd.set_option('precision', 2)

final_df = pd.read_excel('./Lesson_3/GEORG_hyperparamsFit.xlsx') # загружаем результаты  анализа
# final_df = pd.read_excel('source_root/30min_backtest_final.xlsx') # загружаем результаты  анализа
print(final_df)

df_plot = final_df[['val_accuracy', 'dropout', 'n_dense_neurons', 'lr', 'batch_size']]
# df_plot = final_df[['Net Profit [$]', 'buy_before', 'sell_after', 'pattern_size', 'train_window', 'extr_window', 'overlap']]

fig = px.parallel_coordinates(df_plot,
                              color='val_accuracy',
                              range_color=[df_plot['val_accuracy'].min(), df_plot['val_accuracy'].max()],
                              # color="Net Profit [$]",
                              # range_color=[df_plot['Net Profit [$]'].min(), df_plot['Net Profit [$]'].max()],
                              title='Зависимость accuracy от гиперпараметров архитектуры сети и обучения',
                              # color_continuous_scale='Inferno',
                              color_continuous_scale=[
                                  (0.00, 'gray'),   (0.75, 'gray'),
                                  (0.75, 'orange'), (1.00, 'orange')
                              ])

fig.write_html('./Lesson_3/GEORG_hyperparamsFit.html')  # сохраняем в файл
fig.show()

