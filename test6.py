from matplotlib import pyplot as plt
import GPy
import numpy as np

class lvm_visualise:
    def __init__(self, visualise, ax):
        self.cid = ax.figure.canvas.mpl_connect('button_press_event', self.on_click)
        self.cid = ax.figure.canvas.mpl_connect('motion_notify_event', self.on_move)
        self.visualise = visualise
        self.ax = ax
        # This is vectorDisplay code
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
            # This is vectorModify code
            #print 'move', event.xdata, event.ydata
            latent_values = np.array((event.xdata, event.ydata))
            self.visualise.modify(latent_values)
            #print 'y', y

class data_visualiser:
    def __init__(self, model):
        self.model = model

    def modify(self, latent_values):
        raise NotImplementedError, "this needs to be implemented to use the data_visualiser class"

class vector_visualise(data_visualiser):
    def __init__(self, model):
        data_visualiser.__init__(self, model)
        self.fig_display = plt.figure()
        self.model = model
        self.y = model.predict(np.zeros((1, model.input_dim)))[0]
        self.handle = plt.plot(np.arange(0, model.output_dim)[:, None], self.y.T)[0]

    def modify(self, latent_values):
        y = self.model.predict(latent_values)[0]
        xdata, ydata = self.handle.get_data()
        self.handle.set_data(xdata, y.T)
        plt.show()

class image_visualise(data_visualiser):
    def __init__(self, model, dimensions=(16,16), transpose=False, invert=False):
        data_visualiser.__init__(self, model)
        self.fig_display = plt.figure()
        self.model = model
        self.imvals = self.get_array(model, np.zeros((1, model.input_dim)))
        self.handle = plt.imshow(self.imvals)
        self.transpose = transpose
        self.invert = invert

    def modify(self, latent_values):
        xdata, ydata = self.handle.get_data()
        self.imvals = self.get_array(model, latent_values)
        self.handle.set_array(self.imvals)
        plt.show()

    def get_array(self, latent_values):
        self.y = self.model.predict(latent_values)[0]
        imvals = np.reshape(self.y, dimensions)
        if self.transpose:
            imvals = imvals.T
        if self.invert:
            imvals = -imvals

model = GPy.examples.dimensionality_reduction.oil_100()
visualise = vector_visualise(model)
fig = plt.figure()
ax = fig.add_subplot(111)
plt.ylim((-1, 1))
plt.xlim((-1, 1))

ax.set_title('latent space')

linebuilder = lvm_visualise(visualise, ax)

plt.show()
