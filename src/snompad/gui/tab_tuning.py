# tab for the SNOMpad GUI to acquire, demod, and plot parameters concerning
# experiment performance, calibration, or tuning

from bokeh.layouts import layout, column, row


# CALLLBACKS ###########################################################################################################
def update_tuning_tab(buffer):
    pass

# WIDGETS ##############################################################################################################


# LAYOUT ###############################################################################################################
tuning_layout = column([])

# def setup_phase_plot():
#     init_data = {'binned': [np.random.uniform(size=(64, 64))]}
#     plot_data = ColumnDataSource(init_data)
#     fig = figure(height=400, width=400, toolbar_location=None)
#     cmap = LinearColorMapper(palette=cc.bgy, nan_color='#FF0000')
#     plot = fig.image(image='binned', source=plot_data, color_mapper=cmap,
#                      dh=2*np.pi, dw=2*np.pi, x=-np.pi, y=-np.pi, syncable=False)
#     cbar = plot.construct_color_bar(padding=1)
#     fig.add_layout(cbar, 'right')
#     return fig, plot_data
