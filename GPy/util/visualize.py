import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import GPy
import numpy as np
import matplotlib as mpl
import time
import Image
#import visual

class data_show:
    """
    The data_show class is a base class which describes how to visualize a
    particular data set. For example, motion capture data can be plotted as a
    stick figure, or images are shown using imshow. This class enables latent
    to data visualizations for the GP-LVM.
    """
    def __init__(self, vals):
        self.vals = vals.copy()
        # If no axes are defined, create some.

    def modify(self, vals):
        raise NotImplementedError, "this needs to be implemented to use the data_show class"

    def close(self):
        raise NotImplementedError, "this needs to be implemented to use the data_show class"


class vpython_show(data_show):
    """
    the vpython_show class is a base class for all visualization methods that use vpython to display. It is initialized with a scene. If the scene is set to None it creates a scene window.
    """

    def __init__(self, vals, scene=None):
        data_show.__init__(self, vals)
        # If no axes are defined, create some.

        if scene==None:
            self.scene = visual.display(title='Data Visualization')
        else:
            self.scene = scene

    def close(self):
        self.scene.exit()



class matplotlib_show(data_show):
    """
    the matplotlib_show class is a base class for all visualization methods that use matplotlib. It is initialized with an axis. If the axis is set to None it creates a figure window.
    """
    def __init__(self, vals, axes=None):
        data_show.__init__(self, vals)
        # If no axes are defined, create some.

        if axes==None:
            fig = plt.figure()
            self.axes = fig.add_subplot(111)
        else:
            self.axes = axes

    def close(self):
        plt.close(self.axes.get_figure())

class vector_show(matplotlib_show):
    """
    A base visualization class that just shows a data vector as a plot of
    vector elements alongside their indices.
    """
    def __init__(self, vals, axes=None):
        matplotlib_show.__init__(self, vals, axes)
        self.handle = self.axes.plot(np.arange(0, len(vals))[:, None], self.vals.T)[0]

    def modify(self, vals):
        self.vals = vals.copy()
        xdata, ydata = self.handle.get_data()
        self.handle.set_data(xdata, self.vals.T)
        self.axes.figure.canvas.draw()


class lvm(matplotlib_show):
    def __init__(self, vals, model, data_visualize, latent_axes=None, sense_axes=None, latent_index=[0,1]):
        """Visualize a latent variable model

        :param model: the latent variable model to visualize.
        :param data_visualize: the object used to visualize the data which has been modelled.
        :type data_visualize: visualize.data_show  type.
        :param latent_axes: the axes where the latent visualization should be plotted.
        """
        if vals == None:
            vals = model.X[0]

        matplotlib_show.__init__(self, vals, axes=latent_axes)

        if isinstance(latent_axes,mpl.axes.Axes):
            self.cid = latent_axes.figure.canvas.mpl_connect('button_press_event', self.on_click)
            self.cid = latent_axes.figure.canvas.mpl_connect('motion_notify_event', self.on_move)
            self.cid = latent_axes.figure.canvas.mpl_connect('axes_leave_event', self.on_leave)
            self.cid = latent_axes.figure.canvas.mpl_connect('axes_enter_event', self.on_enter)
        else:
            self.cid = latent_axes[0].figure.canvas.mpl_connect('button_press_event', self.on_click)
            self.cid = latent_axes[0].figure.canvas.mpl_connect('motion_notify_event', self.on_move)
            self.cid = latent_axes[0].figure.canvas.mpl_connect('axes_leave_event', self.on_leave)
            self.cid = latent_axes[0].figure.canvas.mpl_connect('axes_enter_event', self.on_enter)

        self.data_visualize = data_visualize
        self.Model = model
        self.latent_axes = latent_axes
        self.sense_axes = sense_axes
        self.called = False
        self.move_on = False
        self.latent_index = latent_index
        self.latent_dim = model.input_dim

        # The red cross which shows current latent point.
        self.latent_values = vals
        self.latent_handle = self.latent_axes.plot([0],[0],'rx',mew=2)[0]
        self.modify(vals)
        self.show_sensitivities()

    def modify(self, vals):
        """When latent values are modified update the latent representation and ulso update the output visualization."""
        self.vals = vals.copy()
        y = self.Model.predict(self.vals)[0]
        self.data_visualize.modify(y)
        self.latent_handle.set_data(self.vals[self.latent_index[0]], self.vals[self.latent_index[1]])
        self.axes.figure.canvas.draw()


    def on_enter(self,event):
        pass
    def on_leave(self,event):
        pass

    def on_click(self, event):
        if event.inaxes!=self.latent_axes: return
        self.move_on = not self.move_on
        self.called = True

    def on_move(self, event):
        if event.inaxes!=self.latent_axes: return
        if self.called and self.move_on:
            # Call modify code on move
            self.latent_values[self.latent_index[0]]=event.xdata
            self.latent_values[self.latent_index[1]]=event.ydata
            self.modify(self.latent_values)

    def show_sensitivities(self):
        # A click in the bar chart axis for selection a dimension.
        if self.sense_axes != None:
            self.sense_axes.cla()
            self.sense_axes.bar(np.arange(self.Model.input_dim),1./self.Model.input_sensitivity(),color='b')

            if self.latent_index[1] == self.latent_index[0]:
                self.sense_axes.bar(np.array(self.latent_index[0]),1./self.Model.input_sensitivity()[self.latent_index[0]],color='y')
                self.sense_axes.bar(np.array(self.latent_index[1]),1./self.Model.input_sensitivity()[self.latent_index[1]],color='y')

            else:
                self.sense_axes.bar(np.array(self.latent_index[0]),1./self.Model.input_sensitivity()[self.latent_index[0]],color='g')
                self.sense_axes.bar(np.array(self.latent_index[1]),1./self.Model.input_sensitivity()[self.latent_index[1]],color='r')

            self.sense_axes.figure.canvas.draw()


class lvm_subplots(lvm):
    """
    latent_axes is a np array of dimension np.ceil(input_dim/2),
    one for each pair of the latent dimensions.
    """
    def __init__(self, vals, Model, data_visualize, latent_axes=None, sense_axes=None):
        self.nplots = int(np.ceil(Model.input_dim/2.))+1
        assert len(latent_axes)==self.nplots
        if vals==None:
            vals = Model.X[0, :]
        self.latent_values = vals 

        for i, axis in enumerate(latent_axes):
            if i == self.nplots-1:
                if self.nplots*2!=Model.input_dim:
                    latent_index = [i*2, i*2]
                lvm.__init__(self, self.latent_vals, Model, data_visualize, axis, sense_axes, latent_index=latent_index)
            else:
                latent_index = [i*2, i*2+1]
                lvm.__init__(self, self.latent_vals, Model, data_visualize, axis, latent_index=latent_index)



class lvm_dimselect(lvm):
    """
    A visualizer for latent variable models which allows selection of the latent dimensions to use by clicking on a bar chart of their length scales.

    For an example of the visualizer's use try:
    
    GPy.examples.dimensionality_reduction.BGPVLM_oil()

    """
    def __init__(self, vals, Model, data_visualize, latent_axes=None, sense_axes=None, latent_index=[0, 1], labels=None):
        if latent_axes==None and sense_axes==None:
            self.fig,(latent_axes,self.sense_axes) = plt.subplots(1,2)
        elif sense_axes==None:
            fig=plt.figure()
            self.sense_axes = fig.add_subplot(111)
        else:
            self.sense_axes = sense_axes
        self.labels = labels
        lvm.__init__(self,vals,Model,data_visualize,latent_axes,sense_axes,latent_index)
        print "use left and right mouse butons to select dimensions"


    def on_click(self, event):

        if event.inaxes==self.sense_axes:
            new_index = max(0,min(int(np.round(event.xdata-0.5)),self.Model.input_dim-1))
            if event.button == 1:
                # Make it red if and y-axis (red=port=left) if it is a left button click
                self.latent_index[1] = new_index                
            else:
                # Make it green and x-axis (green=starboard=right) if it is a right button click
                self.latent_index[0] = new_index
                
            self.show_sensitivities()

            self.latent_axes.cla()
            self.Model.plot_latent(which_indices=self.latent_index,
                                   ax=self.latent_axes, labels=self.labels)
            self.latent_handle = self.latent_axes.plot([0],[0],'rx',mew=2)[0]
            self.modify(self.latent_values)

        elif event.inaxes==self.latent_axes:
            self.move_on = not self.move_on

        self.called = True



    def on_leave(self,event):
        latent_values = self.latent_values.copy()
        y = self.Model.predict(latent_values[None,:])[0]
        self.data_visualize.modify(y)



class image_show(matplotlib_show):
    """Show a data vector as an image."""
    def __init__(self, vals, axes=None, dimensions=(16,16), transpose=False, invert=False, scale=False, palette=[], presetMean = 0., presetSTD = -1., selectImage=0):
        matplotlib_show.__init__(self, vals, axes)
        self.dimensions = dimensions
        self.transpose = transpose
        self.invert = invert
        self.scale = scale
        self.palette = palette
        self.presetMean = presetMean
        self.presetSTD = presetSTD
        self.selectImage = selectImage # This is used when the y vector contains multiple images concatenated.

        self.set_image(self.vals)
        if not self.palette == []: # Can just show the image (self.set_image() took care of setting the palette)
            self.handle = self.axes.imshow(self.vals, interpolation='nearest')
        else: # Use a boring gray map.
            self.handle = self.axes.imshow(self.vals, cmap=plt.cm.gray, interpolation='nearest') # @UndefinedVariable
        plt.show()

    def modify(self, vals):
        self.set_image(vals.copy())
        self.handle.set_array(self.vals)
        self.axes.figure.canvas.draw() 

    def set_image(self, vals):
        dim = self.dimensions[0] * self.dimensions[1]
        nImg = np.sqrt(vals[0,].size/dim)
        if nImg > 1 and nImg.is_integer(): # Show a mosaic of images
            nImg = np.int(nImg)
            self.vals = np.zeros((self.dimensions[0]*nImg, self.dimensions[1]*nImg))
            for iR in range(nImg):
                for iC in range(nImg):
                    currImgId = iR*nImg + iC
                    currImg = np.reshape(vals[0,dim*currImgId+np.array(range(dim))], self.dimensions, order='F')
                    firstRow = iR*self.dimensions[0]
                    lastRow = (iR+1)*self.dimensions[0]
                    firstCol = iC*self.dimensions[1]
                    lastCol = (iC+1)*self.dimensions[1]
                    self.vals[firstRow:lastRow, firstCol:lastCol] = currImg

        else: 
            self.vals = np.reshape(vals[0,dim*self.selectImage+np.array(range(dim))], self.dimensions, order='F')
        if self.transpose:
            self.vals = self.vals.T
        # if not self.scale:
        #     self.vals = self.vals
        if self.invert:
            self.vals = -self.vals

        # un-normalizing, for visualisation purposes:
        if self.presetSTD >= 0: # The Mean is assumed to be in the range (0,255)
            self.vals = self.vals*self.presetSTD + self.presetMean
            # Clipping the values:
            self.vals[self.vals < 0] = 0
            self.vals[self.vals > 255] = 255
        else:
            self.vals = 255*(self.vals - self.vals.min())/(self.vals.max() - self.vals.min())
        if not self.palette == []: # applying using an image palette (e.g. if the image has been quantized)
            self.vals = Image.fromarray(self.vals.astype('uint8'))
            self.vals.putpalette(self.palette) # palette is a list, must be loaded before calling this function

class mocap_data_show_vpython(vpython_show):
    """Base class for visualizing motion capture data using visual module."""

    def __init__(self, vals, scene=None, connect=None, radius=0.1):
        vpython_show.__init__(self, vals, scene)
        self.radius = radius
        self.connect = connect
        self.process_values()
        self.draw_edges()
        self.draw_vertices()

    def draw_vertices(self):
        self.spheres = []
        for i in range(self.vals.shape[0]):
            self.spheres.append(visual.sphere(pos=(self.vals[i, 0], self.vals[i, 2], self.vals[i, 1]), radius=self.radius))
        self.scene.visible=True
        
    def draw_edges(self):
        self.rods = []
        self.line_handle = []
        if not self.connect==None:
            self.I, self.J = np.nonzero(self.connect)
            for i, j in zip(self.I, self.J):
                pos, axis = self.pos_axis(i, j)
                self.rods.append(visual.cylinder(pos=pos, axis=axis, radius=self.radius))

    def modify_vertices(self):
        for i in range(self.vals.shape[0]):
            self.spheres[i].pos = (self.vals[i, 0], self.vals[i, 2], self.vals[i, 1])

    def modify_edges(self):
        self.line_handle = []
        if not self.connect==None:            
            self.I, self.J = np.nonzero(self.connect)
            for rod, i, j in zip(self.rods, self.I, self.J):
                rod.pos, rod.axis = self.pos_axis(i, j)

    def pos_axis(self, i, j):
        pos = []
        axis = []
        pos.append(self.vals[i, 0])
        axis.append(self.vals[j, 0]-self.vals[i,0])
        pos.append(self.vals[i, 2])
        axis.append(self.vals[j, 2]-self.vals[i,2])
        pos.append(self.vals[i, 1])
        axis.append(self.vals[j, 1]-self.vals[i,1])
        return pos, axis

    def modify(self, vals):
        self.vals = vals.copy()
        self.process_values()
        self.modify_edges()
        self.modify_vertices()

    def process_values(self):
        raise NotImplementedError, "this needs to be implemented to use the data_show class"


class mocap_data_show(matplotlib_show):
    """Base class for visualizing motion capture data."""

    def __init__(self, vals, axes=None, connect=None):
        if axes==None:
            fig = plt.figure()
            axes = fig.add_subplot(111, projection='3d')
        matplotlib_show.__init__(self, vals, axes)

        self.connect = connect
        self.process_values()
        self.initialize_axes()
        self.draw_vertices()
        self.finalize_axes()
        self.draw_edges()
        self.axes.figure.canvas.draw()

    def draw_vertices(self):
        self.points_handle = self.axes.scatter(self.vals[:, 0], self.vals[:, 1], self.vals[:, 2])
        
    def draw_edges(self):
        self.line_handle = []
        if not self.connect==None:
            x = []
            y = []
            z = []
            self.I, self.J = np.nonzero(self.connect)
            for i, j in zip(self.I, self.J):
                x.append(self.vals[i, 0])
                x.append(self.vals[j, 0])
                x.append(np.NaN)
                y.append(self.vals[i, 1])
                y.append(self.vals[j, 1])
                y.append(np.NaN)
                z.append(self.vals[i, 2])
                z.append(self.vals[j, 2])
                z.append(np.NaN)
            self.line_handle = self.axes.plot(np.array(x), np.array(y), np.array(z), 'b-')
            
    def modify(self, vals):
        self.vals = vals.copy()
        self.process_values()
        self.initialize_axes_modify()
        self.draw_vertices()
        self.finalize_axes_modify()
        self.draw_edges()
        self.axes.figure.canvas.draw()

    def process_values(self):
        raise NotImplementedError, "this needs to be implemented to use the data_show class"

    def initialize_axes(self):
        """Set up the axes with the right limits and scaling."""
        self.x_lim = np.array([self.vals[:, 0].min(), self.vals[:, 0].max()])
        self.y_lim = np.array([self.vals[:, 1].min(), self.vals[:, 1].max()])
        self.z_lim = np.array([self.vals[:, 2].min(), self.vals[:, 2].max()])

    def initialize_axes_modify(self):
        self.points_handle.remove()
        self.line_handle[0].remove()

    def finalize_axes(self):
        self.axes.set_xlim(self.x_lim)
        self.axes.set_ylim(self.y_lim)
        self.axes.set_zlim(self.z_lim)
        self.axes.auto_scale_xyz([-1., 1.], [-1., 1.], [-1.5, 1.5])
        
        #self.axes.set_aspect('equal')
        self.axes.autoscale(enable=False)

    def finalize_axes_modify(self):
        self.axes.set_xlim(self.x_lim)
        self.axes.set_ylim(self.y_lim)
        self.axes.set_zlim(self.z_lim)


class stick_show(mocap_data_show_vpython):
    """Show a three dimensional point cloud as a figure. Connect elements of the figure together using the matrix connect."""
    def __init__(self, vals, connect=None, scene=None):
        mocap_data_show_vpython.__init__(self, vals, scene=scene, connect=connect, radius=0.04)

    def process_values(self):
        self.vals = self.vals.reshape((3, self.vals.shape[1]/3)).T

class skeleton_show(mocap_data_show_vpython):
    """data_show class for visualizing motion capture data encoded as a skeleton with angles."""
    def __init__(self, vals, skel, scene=None, padding=0):
        """data_show class for visualizing motion capture data encoded as a skeleton with angles.
        :param vals: set of modeled angles to use for printing in the axis when it's first created.
        :type vals: np.array
        :param skel: skeleton object that has the parameters of the motion capture skeleton associated with it.
        :type skel: mocap.skeleton object 
        :param padding:
        :type int
        """
        self.skel = skel
        self.padding = padding
        connect = skel.connection_matrix()
        mocap_data_show_vpython.__init__(self, vals, scene=scene, connect=connect, radius=0.4)
    def process_values(self):
        """Takes a set of angles and converts them to the x,y,z coordinates in the internal prepresentation of the class, ready for plotting.

        :param vals: the values that are being modelled."""
        
        if self.padding>0:
            channels = np.zeros((self.vals.shape[0], self.vals.shape[1]+self.padding))
            channels[:, 0:self.vals.shape[0]] = self.vals
        else:
            channels = self.vals
        vals_mat = self.skel.to_xyz(channels.flatten())
        self.vals = np.zeros_like(vals_mat)
        # Flip the Y and Z axes
        self.vals[:, 0] = vals_mat[:, 0].copy()
        self.vals[:, 1] = vals_mat[:, 2].copy()
        self.vals[:, 2] = vals_mat[:, 1].copy()
        
    def wrap_around(self, lim, connect):
        quot = lim[1] - lim[0]
        self.vals = rem(self.vals, quot)+lim[0]
        nVals = floor(self.vals/quot)
        for i in range(connect.shape[0]):
            for j in find(connect[i, :]):
                if nVals[i] != nVals[j]:
                    connect[i, j] = False
        return connect


def data_play(Y, visualizer, frame_rate=30):
    """Play a data set using the data_show object given.

    :Y: the data set to be visualized.
    :param visualizer: the data show objectwhether to display during optimisation
    :type visualizer: data_show

    Example usage:

    This example loads in the CMU mocap database (http://mocap.cs.cmu.edu) subject number 35 motion number 01. It then plays it using the mocap_show visualize object.
    
    data = GPy.util.datasets.cmu_mocap(subject='35', train_motions=['01'])
    Y = data['Y']
    Y[:, 0:3] = 0.   # Make figure walk in place
    visualize = GPy.util.visualize.skeleton_show(Y[0, :], data['skel'])
    GPy.util.visualize.data_play(Y, visualize)
    """
    

    for y in Y:
        visualizer.modify(y)
        time.sleep(1./float(frame_rate))
