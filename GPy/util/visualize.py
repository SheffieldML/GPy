import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import GPy
import numpy as np

class lvm:
    def __init__(self, model, data_visualize, ax):
        self.cid = ax.figure.canvas.mpl_connect('button_press_event', self.on_click)
        self.cid = ax.figure.canvas.mpl_connect('motion_notify_event', self.on_move)
        self.data_visualize = data_visualize
        self.model = model
        self.ax = ax
        self.called = False
        self.move_on = False

    def on_click(self, event):
        print 'click', event.xdata, event.ydata
        if event.inaxes!=self.ax: return
        self.move_on = not self.move_on
        print
        if self.called:
            self.xs.append(event.xdata)
            self.ys.append(event.ydata)
            self.line.set_data(self.xs, self.ys)
            self.line.figure.canvas.draw()
        else:
            self.xs = [event.xdata]
            self.ys = [event.ydata]
            self.line, = ax.plot(event.xdata, event.ydata)
        self.called = True
    def on_move(self, event):
        if event.inaxes!=self.ax: return
        if self.called and self.move_on:
            # Call modify code on move
            #print 'move', event.xdata, event.ydata
            latent_values = np.array((event.xdata, event.ydata))
            y = self.model.predict(latent_values)[0]
            self.data_visualize.modify(y)
            #print 'y', y

class data_show:
    """The data show class is a base class which describes how to visualize a particular data set. For example, motion capture data can be plotted as a stick figure, or images are shown using imshow. This class enables latent to data visualizations for the GP-LVM."""

    def __init__(self, vals, axis=None):
        self.vals = vals
        # If no axes are defined, create some.
        if axis==None:
            fig = plt.figure()
            self.axis = fig.add_subplot(111)            
        else:
            self.axis = axis

    def modify(self, vals):
        raise NotImplementedError, "this needs to be implemented to use the data_show class"

class vector_show(data_show):
    """A base visualization class that just shows a data vector as a plot of vector elements alongside their indices."""
    def __init__(self, vals, axis=None):
        data_show.__init__(self, vals, axis)
        self.vals = vals.T
        self.handle = plt.plot(np.arange(0, len(vals))[:, None], self.vals)[0]

    def modify(self, vals):
        xdata, ydata = self.handle.get_data()
        self.vals = vals.T
        self.handle.set_data(xdata, self.vals)
        self.axis.figure.canvas.draw()

class image_show(data_show):
    """Show a data vector as an image."""
    def __init__(self, vals, axis=None, dimensions=(16,16), transpose=False, invert=False):
        data_show.__init__(self, vals, axis)
        self.dimensions = dimensions
        self.fig_display = plt.figure()
        self.set_image(vals)
        self.handle = plt.imshow(self.vals)
        self.transpose = transpose
        self.invert = invert

    def modify(self, vals):
        self.set_image(vals)
        self.handle.set_array(self.vals)
        self.axis.figure.canvas.draw()

    def set_image(self, vals):
        self.vals = np.reshape(vals, self.dimensions)
        if self.transpose:
            self.vals = self.vals.T
        if self.invert:
            self.vals = -self.vals

class stick_show(data_show):
    """Show a three dimensional point cloud as a figure. Connect elements of the figure together using the matrix connect."""

    def __init__(self, vals, axis=None, connect=None):
        if axis==None:
            fig = plt.figure()
            axis = fig.add_subplot(111, projection='3d')
        data_show.__init__(self, vals, axis)
        self.vals = vals.reshape((3, vals.shape[1]/3)).T
        self.x_lim = np.array([self.vals[:, 0].min(), self.vals[:, 0].max()])
        self.y_lim = np.array([self.vals[:, 1].min(), self.vals[:, 1].max()])
        self.z_lim = np.array([self.vals[:, 2].min(), self.vals[:, 2].max()])
        self.points_handle = self.axis.scatter(self.vals[:, 0], self.vals[:, 1], self.vals[:, 2])
        self.axis.set_xlim(self.x_lim)
        self.axis.set_ylim(self.y_lim)
        self.axis.set_zlim(self.z_lim)
        self.axis.set_aspect(1)
        self.axis.autoscale(enable=False)

        self.connect = connect
        if not self.connect==None:
            x = []
            y = []
            z = []
            self.I, self.J = np.nonzero(self.connect)
            for i in range(len(self.I)):
                x.append(self.vals[self.I[i], 0])
                x.append(self.vals[self.J[i], 0])
                x.append(np.NaN)
                y.append(self.vals[self.I[i], 1])
                y.append(self.vals[self.J[i], 1])
                y.append(np.NaN)
                z.append(self.vals[self.I[i], 2])
                z.append(self.vals[self.J[i], 2])
                z.append(np.NaN)
            self.line_handle = self.axis.plot(np.array(x), np.array(y), np.array(z), 'b-')
        self.axis.figure.canvas.draw()

    def modify(self, vals):
        self.points_handle.remove()
        self.line_handle[0].remove()
        self.vals = vals.reshape((3, vals.shape[1]/3)).T
        self.points_handle = self.axis.scatter(self.vals[:, 0], self.vals[:, 1], self.vals[:, 2])
        self.axis.set_xlim(self.x_lim)
        self.axis.set_ylim(self.y_lim)
        self.axis.set_zlim(self.z_lim)
        self.line_handle = []
        if not self.connect==None:
            x = []
            y = []
            z = []
            self.I, self.J = np.nonzero(self.connect)
            for i in range(len(self.I)):
                x.append(self.vals[self.I[i], 0])
                x.append(self.vals[self.J[i], 0])
                x.append(np.NaN)
                y.append(self.vals[self.I[i], 1])
                y.append(self.vals[self.J[i], 1])
                y.append(np.NaN)
                z.append(self.vals[self.I[i], 2])
                z.append(self.vals[self.J[i], 2])
                z.append(np.NaN)
            self.line_handle = self.axis.plot(np.array(x), np.array(y), np.array(z), 'b-')

        self.axis.figure.canvas.draw()
        


