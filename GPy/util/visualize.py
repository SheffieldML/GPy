import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import GPy
import numpy as np
import matplotlib as mpl

class lvm:
    def __init__(self, model, data_visualize, latent_axes, latent_index=[0,1]):
        if isinstance(latent_axes,mpl.axes.Axes):
            self.cid = latent_axes.figure.canvas.mpl_connect('button_press_event', self.on_click)
            self.cid = latent_axes.figure.canvas.mpl_connect('motion_notify_event', self.on_move)
        else:
            self.cid = latent_axes[0].figure.canvas.mpl_connect('button_press_event', self.on_click)
            self.cid = latent_axes[0].figure.canvas.mpl_connect('motion_notify_event', self.on_move)
        self.data_visualize = data_visualize
        self.model = model
        self.latent_axes = latent_axes

        self.called = False
        self.move_on = False
        self.latent_index = latent_index
        self.latent_dim = model.Q

    def on_click(self, event):
        #print 'click', event.xdata, event.ydata
        if event.inaxes!=self.latent_axes: return
        self.move_on = not self.move_on
        # if self.called:
        #     self.xs.append(event.xdata)
        #     self.ys.append(event.ydata)
        #     self.line.set_data(self.xs, self.ys)
        #     self.line.figure.canvas.draw()
        # else:
        #     self.xs = [event.xdata]
        #     self.ys = [event.ydata]
        #     self.line, = self.latent_axes.plot(event.xdata, event.ydata)
        self.called = True
    def on_move(self, event):
        if event.inaxes!=self.latent_axes: return
        if self.called and self.move_on:
            # Call modify code on move
            #print 'move', event.xdata, event.ydata
            latent_values = np.zeros((1,self.latent_dim))
            latent_values[0,self.latent_index] = np.array([event.xdata, event.ydata])
            y = self.model.predict(latent_values)[0]
            self.data_visualize.modify(y)
            #print 'y', y

class lvm_subplots(lvm):
    """
    latent_axes is a np array of dimension np.ceil(Q/2) + 1,
    one for each pair of the axes, and the last one for the sensitiity histogram
    """
    def __init__(self, model, data_visualize, latent_axes=None, latent_index=[0,1]):
        self.nplots = int(np.ceil(model.Q/2.))+1
        lvm.__init__(self,model,data_visualize,latent_axes,latent_index)
        self.latent_values = np.zeros(2*np.ceil(self.model.Q/2.)) # possibly an extra dimension on this
        assert latent_axes.size == self.nplots


class lvm_dimselect(lvm):
    """
    A visualizer for latent variable models
    with selection by clicking on the histogram
    """
    def __init__(self, model, data_visualize):
        self.fig,(latent_axes,self.hist_axes) = plt.subplots(1,2)

        lvm.__init__(self,model,data_visualize,latent_axes,[0,1])
        self.latent_values_clicked = np.zeros(model.Q)
        self._first_index_next = False

    def on_click(self, event):
        #print "click"
        if event.inaxes==self.hist_axes:
            self.hist_axes.cla()
            self.hist_axes.bar(np.arange(self.model.Q),1./self.model.input_sensitivity(),color='b')
            new_index = int(np.round(event.xdata))
            self.latent_index[int(self._first_index_next)] = new_index
            self._first_index_next = not self._first_index_next
            self.hist_axes.bar(np.array(self.latent_index),1./self.model.input_sensitivity()[self.latent_index],color='r')
            self.latent_axes.cla()
            self.model.plot_latent(which_indices = self.latent_index,ax=self.latent_axes)
            self.fig.canvas.draw()
        if event.inaxes==self.latent_axes:
            #self.latent_values_clicked[self.latent_index] = np.array([event.xdata,event.ydata])
            pass
        self.move_on=True
        self.called = True


    def on_move(self, event):
        #print "move"
        if event.inaxes!=self.latent_axes: return
        if self.called and self.move_on:
            latent_values = self.latent_values_clicked.copy()
            latent_values[self.latent_index] = np.array([event.xdata, event.ydata])
            y = self.model.predict(latent_values[None,:])[0]
            self.data_visualize.modify(y)



class data_show:
    """
    The data show class is a base class which describes how to visualize a
    particular data set. For example, motion capture data can be plotted as a
    stick figure, or images are shown using imshow. This class enables latent
    to data visualizations for the GP-LVM.
    """

    def __init__(self, vals, axes=None):
        self.vals = vals
        # If no axes are defined, create some.
        if axes==None:
            fig = plt.figure()
            self.axes = fig.add_subplot(111)
        else:
            self.axes = axes

    def modify(self, vals):
        raise NotImplementedError, "this needs to be implemented to use the data_show class"

class vector_show(data_show):
    """
    A base visualization class that just shows a data vector as a plot of
    vector elements alongside their indices.
    """
    def __init__(self, vals, axes=None):
        data_show.__init__(self, vals, axes)
        self.vals = vals.T
        self.handle = self.axes.plot(np.arange(0, len(vals))[:, None], self.vals)[0]

    def modify(self, vals):
        xdata, ydata = self.handle.get_data()
        self.vals = vals.T
        self.handle.set_data(xdata, self.vals)
        self.axes.figure.canvas.draw()

class image_show(data_show):
    """Show a data vector as an image."""
    def __init__(self, vals, axes=None, dimensions=(16,16), transpose=False, invert=False, scale=False):
        data_show.__init__(self, vals, axes)
        self.dimensions = dimensions
        self.transpose = transpose
        self.invert = invert
        self.scale = scale
        self.set_image(vals/255.)
        self.handle = self.axes.imshow(self.vals, cmap=plt.cm.gray, interpolation='nearest')
        plt.show()

    def modify(self, vals):
        self.set_image(vals/255.)
        #self.handle.remove()
        #self.handle = self.axes.imshow(self.vals)
        self.handle.set_array(self.vals)
        #self.axes.figure.canvas.draw()
        plt.show()

    def set_image(self, vals):
        self.vals = np.reshape(vals, self.dimensions, order='F')
        if self.transpose:
            self.vals = self.vals.T
        if not self.scale:
            self.vals = self.vals
        #if self.invert:
        #    self.vals = -self.vals

class stick_show(data_show):
    """Show a three dimensional point cloud as a figure. Connect elements of the figure together using the matrix connect."""

    def __init__(self, vals, axes=None, connect=None):
        if axes==None:
            fig = plt.figure()
            axes = fig.add_subplot(111, projection='3d')
        data_show.__init__(self, vals, axes)
        self.vals = vals.reshape((3, vals.shape[1]/3)).T
        self.x_lim = np.array([self.vals[:, 0].min(), self.vals[:, 0].max()])
        self.y_lim = np.array([self.vals[:, 1].min(), self.vals[:, 1].max()])
        self.z_lim = np.array([self.vals[:, 2].min(), self.vals[:, 2].max()])
        self.points_handle = self.axes.scatter(self.vals[:, 0], self.vals[:, 1], self.vals[:, 2])
        self.axes.set_xlim(self.x_lim)
        self.axes.set_ylim(self.y_lim)
        self.axes.set_zlim(self.z_lim)
        self.axes.set_aspect(1)
        self.axes.autoscale(enable=False)

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
            self.line_handle = self.axes.plot(np.array(x), np.array(y), np.array(z), 'b-')
        self.axes.figure.canvas.draw()

    def modify(self, vals):
        self.points_handle.remove()
        self.line_handle[0].remove()
        self.vals = vals.reshape((3, vals.shape[1]/3)).T
        self.points_handle = self.axes.scatter(self.vals[:, 0], self.vals[:, 1], self.vals[:, 2])
        self.axes.set_xlim(self.x_lim)
        self.axes.set_ylim(self.y_lim)
        self.axes.set_zlim(self.z_lim)
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
            self.line_handle = self.axes.plot(np.array(x), np.array(y), np.array(z), 'b-')

        self.axes.figure.canvas.draw()



