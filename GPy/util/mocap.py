import os
import numpy as np
import math
from GPy.util import datasets as dat

class vertex:
    def __init__(self, name, id, parents=[], children=[], meta = {}):
        self.name = name
        self.id = id
        self.parents = parents
        self.children = children
        self.meta = meta

    def __str__(self):
        return self.name + '(' + str(self.id) + ').'
        
class tree:
    def __init__(self):
        self.vertices = []
        self.vertices.append(vertex(name='root', id=0))

    def __str__(self):
        index = self.find_root()
        return self.branch_str(index)

    def branch_str(self, index, indent=''):
        out = indent + str(self.vertices[index]) + '\n'
        for child in self.vertices[index].children:
            out+=self.branch_str(child, indent+'  ')
        return out

    def find_children(self):
        """Take a tree and set the children according to the parents.

        Takes a tree structure which lists the parents of each vertex
        and computes the children for each vertex and places them in."""
        for i in range(len(self.vertices)):
            self.vertices[i].children = []
        for i in range(len(self.vertices)):
            for parent in self.vertices[i].parents:
                if i not in self.vertices[parent].children:
                    self.vertices[parent].children.append(i) 

    def find_parents(self):
        """Take a tree and set the parents according to the children

        Takes a tree structure which lists the children of each vertex
        and computes the parents for each vertex and places them in."""
        for i in range(len(self.vertices)):
            self.vertices[i].parents = []
        for i in range(len(self.vertices)):
            for child in self.vertices[i].children:
                if i not in self.vertices[child].parents:
                    self.vertices[child].parents.append(i) 
                    
    def find_root(self):
        """Finds the index of the root node of the tree."""
        self.find_parents()
        index = 0
        while len(self.vertices[index].parents)>0:
            index = self.vertices[index].parents[0]
        return index
            
    def get_index_by_id(self, id):
        """Give the index associated with a given vertex id."""
        for i in range(len(self.vertices)):
            if self.vertices[i].id == id:
                return i
        raise ValueError('Reverse look up of id failed.')

    def get_index_by_name(self, name):
        """Give the index associated with a given vertex name."""
        for i in range(len(self.vertices)):
            if self.vertices[i].name == name:
                return i
        raise ValueError('Reverse look up of name failed.')

    def order_vertices(self):
        """Order vertices in the graph such that parents always have a lower index than children."""
        
        ordered = False
        while ordered == False:
            for i in range(len(self.vertices)):
                ordered = True
                for parent in self.vertices[i].parents:
                    if parent>i:
                        ordered = False
                        self.swap_vertices(i, parent)




    def swap_vertices(self, i, j):
        """
        Swap two vertices in the tree structure array.
        swap_vertex swaps the location of two vertices in a tree structure array. 

        :param tree: the tree for which two vertices are to be swapped.
        :param i: the index of the first vertex to be swapped.
        :param j: the index of the second vertex to be swapped.
        :rval tree: the tree structure with the two vertex locations swapped.

        """
        store_vertex_i = self.vertices[i]
        store_vertex_j = self.vertices[j]
        self.vertices[j] = store_vertex_i
        self.vertices[i] = store_vertex_j
        for k in range(len(self.vertices)):
            for swap_list in [self.vertices[k].children, self.vertices[k].parents]:
                if i in swap_list:
                    swap_list[swap_list.index(i)] = -1
                if j in swap_list:
                    swap_list[swap_list.index(j)] = i
                if -1 in swap_list:
                    swap_list[swap_list.index(-1)] = j



def rotation_matrix(xangle, yangle, zangle, order='zxy', degrees=False):

    """

    Compute the rotation matrix for an angle in each direction.
    This is a helper function for computing the rotation matrix for a given set of angles in a given order.

    :param xangle: rotation for x-axis.
    :param yangle: rotation for y-axis.
    :param zangle: rotation for z-axis.
    :param order: the order for the rotations.

     """
    if degrees:
        xangle = math.radians(xangle)
        yangle = math.radians(yangle)
        zangle = math.radians(zangle)

    # Here we assume we rotate z, then x then y.
    c1 = math.cos(xangle) # The x angle
    c2 = math.cos(yangle) # The y angle
    c3 = math.cos(zangle) # the z angle
    s1 = math.sin(xangle)
    s2 = math.sin(yangle)
    s3 = math.sin(zangle)

    # see http://en.wikipedia.org/wiki/Rotation_matrix for
    # additional info.

    if order=='zxy':
        rot_mat = np.array([[c2*c3-s1*s2*s3, c2*s3+s1*s2*c3, -s2*c1],[-c1*s3, c1*c3, s1],[s2*c3+c2*s1*s3, s2*s3-c2*s1*c3, c2*c1]])
    else:
        rot_mat = np.eye(3)
        for i in range(len(order)):
            if order[i]=='x':
                rot_mat = np.dot(np.array([[1, 0, 0], [0,  c1, s1], [0, -s1, c1]]),rot_mat)
            elif order[i] == 'y':
                rot_mat = np.dot(np.array([[c2, 0, -s2], [0, 1, 0], [s2, 0, c2]]),rot_mat)
            elif order[i] == 'z':
                rot_mat = np.dot(np.array([[c3, s3, 0], [-s3, c3, 0], [0, 0, 1]]),rot_mat)

    return rot_mat


# Motion capture data routines.
class skeleton(tree):
    def __init__(self):
        super(skeleton, self).__init__()

    def connection_matrix(self):
        connection = np.zeros((len(self.vertices), len(self.vertices)), dtype=bool)
        for i in range(len(self.vertices)):
            for j in range(len(self.vertices[i].children)):
                connection[i, self.vertices[i].children[j]] = True
        return connection

    def to_xyz(self, channels):
        raise NotImplementedError("this needs to be implemented to use the skeleton class")


    def finalize(self):
        """After loading in a skeleton ensure parents are correct, vertex orders are correct and rotation matrices are correct."""

        self.find_parents()
        self.order_vertices()
        self.set_rotation_matrices()

    def smooth_angle_channels(self, channels):
        """Remove discontinuities in angle channels so that they don't cause artifacts in algorithms that rely on the smoothness of the functions."""
        for vertex in self.vertices:
            for col in vertex.meta['rot_ind']:
                if col:
                    for k in range(1, channels.shape[0]):
                        diff=channels[k, col]-channels[k-1, col]
                    if abs(diff+360.)<abs(diff):
                        channels[k:, col]=channels[k:, col]+360.
                    elif abs(diff-360.)<abs(diff):
                        channels[k:, col]=channels[k:, col]-360.

# class bvh_skeleton(skeleton):
#     def __init__(self):
#         super(bvh_skeleton, self).__init__()

#     def to_xyz(self, channels):
        
class acclaim_skeleton(skeleton):
    def __init__(self, file_name=None):
        super(acclaim_skeleton, self).__init__()
        self.documentation = []
        self.angle = 'deg'
        self.length = 1.0
        self.mass = 1.0
        self.type = 'acclaim'
        self.vertices[0] = vertex(name='root', id=0,
                             parents = [0], children=[],
                             meta = {'orientation': [], 
                                     'axis': [0., 0., 0.], 
                                     'axis_order': [], 
                                     'C': np.eye(3), 
                                     'Cinv': np.eye(3), 
                                     'channels': [], 
                                     'bodymass': [], 
                                     'confmass': [], 
                                     'order': [], 
                                     'rot_ind': [], 
                                     'pos_ind': [], 
                                     'limits': [],
                                     'xyz': np.array([0., 0., 0.]),
                                     'rot': np.eye(3)})

        if file_name:
            self.load_skel(file_name)

    def to_xyz(self, channels):
        rot_val = list(self.vertices[0].meta['orientation'])
        for i in range(len(self.vertices[0].meta['rot_ind'])):
            rind = self.vertices[0].meta['rot_ind'][i]
            if rind != -1:
                rot_val[i] += channels[rind]

        self.vertices[0].meta['rot'] = rotation_matrix(rot_val[0],
                                                       rot_val[1],
                                                       rot_val[2],
                                                       self.vertices[0].meta['axis_order'],
                                                       degrees=True)
        # vertex based store of the xyz location
        self.vertices[0].meta['xyz'] = list(self.vertices[0].meta['offset'])

        for i in range(len(self.vertices[0].meta['pos_ind'])):
            pind = self.vertices[0].meta['pos_ind'][i]
            if pind != -1:
                self.vertices[0].meta['xyz'][i] += channels[pind]


        for i in range(len(self.vertices[0].children)):
            ind = self.vertices[0].children[i]
            self.get_child_xyz(ind, channels)

        xyz = []
        for vertex in self.vertices:
            xyz.append(vertex.meta['xyz'])
        return np.array(xyz)



    def get_child_xyz(self, ind, channels):

        parent = self.vertices[ind].parents[0]
        children = self.vertices[ind].children
        rot_val = np.zeros(3)
        for j in range(len(self.vertices[ind].meta['rot_ind'])):
            rind = self.vertices[ind].meta['rot_ind'][j]
            if rind != -1:
                rot_val[j] = channels[rind]
            else:
                rot_val[j] = 0
        tdof = rotation_matrix(rot_val[0], rot_val[1], rot_val[2],
                               self.vertices[ind].meta['order'],
                               degrees=True)

        torient = rotation_matrix(self.vertices[ind].meta['axis'][0],
                                  self.vertices[ind].meta['axis'][1],
                                  self.vertices[ind].meta['axis'][2],
                                  self.vertices[ind].meta['axis_order'],
                                  degrees=True)

        torient_inv = rotation_matrix(-self.vertices[ind].meta['axis'][0],
                                      -self.vertices[ind].meta['axis'][1],
                                      -self.vertices[ind].meta['axis'][2],
                                      self.vertices[ind].meta['axis_order'][::-1],
                                      degrees=True)

        self.vertices[ind].meta['rot'] = np.dot(np.dot(np.dot(torient_inv,tdof),torient),self.vertices[parent].meta['rot'])


        self.vertices[ind].meta['xyz'] = self.vertices[parent].meta['xyz'] + np.dot(self.vertices[ind].meta['offset'],self.vertices[ind].meta['rot'])

        for i in range(len(children)):
            cind = children[i]
            self.get_child_xyz(cind, channels)


    def load_channels(self, file_name):

        fid=open(file_name, 'r')
        channels = self.read_channels(fid)
        fid.close()
        return channels
    
    def save_channels(self, file_name, channels):
        with open(file_name,'w') as fid:
            self.writ_channels(fid, channels)
            fid.close()

    def load_skel(self, file_name):

        """
        Loads an ASF file into a skeleton structure.

        :param file_name: The file name to load in.

         """         

        fid = open(file_name, 'r')
        self.read_skel(fid)
        fid.close()
        self.name = file_name


    def read_bonedata(self, fid):
        """Read bone data from an acclaim skeleton file stream."""

        bone_count = 0
        lin = self.read_line(fid)
        while lin[0]!=':':
            parts = lin.split()
            if parts[0] == 'begin':
                bone_count += 1
                self.vertices.append(vertex(name = '', id=np.NaN,
                                       meta={'name': [],
                                             'id': [], 
                                             'offset': [], 
                                             'orientation': [], 
                                             'axis': [0., 0., 0.], 
                                             'axis_order': [], 
                                             'C': np.eye(3), 
                                             'Cinv': np.eye(3), 
                                             'channels': [], 
                                             'bodymass': [], 
                                             'confmass': [], 
                                             'order': [], 
                                             'rot_ind': [], 
                                             'pos_ind': [], 
                                             'limits': [],
                                             'xyz': np.array([0., 0., 0.]),
                                             'rot': np.eye(3)}))
                lin = self.read_line(fid)


            elif parts[0]=='id':
                self.vertices[bone_count].id = int(parts[1])
                lin = self.read_line(fid)

                self.vertices[bone_count].children = []

            elif parts[0]=='name':
                self.vertices[bone_count].name = parts[1]
                lin = self.read_line(fid)


            elif parts[0]=='direction':
                direction = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
                lin = self.read_line(fid)


            elif parts[0]=='length':
                lgth =  float(parts[1])
                lin = self.read_line(fid)


            elif parts[0]=='axis':
                self.vertices[bone_count].meta['axis'] = np.array([float(parts[1]),
                                                         float(parts[2]),
                                                         float(parts[3])])
                # order is reversed compared to bvh
                self.vertices[bone_count].meta['axis_order'] =  parts[-1][::-1].lower()
                lin = self.read_line(fid)

            elif parts[0]=='dof':
                order = []
                for i in range(1, len(parts)):
                    if parts[i]== 'rx':
                        chan = 'Xrotation'
                        order.append('x')
                    elif parts[i] =='ry':
                        chan = 'Yrotation'
                        order.append('y')
                    elif parts[i] == 'rz':
                        chan = 'Zrotation'
                        order.append('z')
                    elif parts[i] == 'tx':
                        chan = 'Xposition'
                    elif parts[i] == 'ty':
                        chan = 'Yposition'
                    elif parts[i] == 'tz':
                        chan = 'Zposition'
                    elif parts[i] == 'l':
                        chan = 'length'
                    self.vertices[bone_count].meta['channels'].append(chan)
                    # order is reversed compared to bvh
                self.vertices[bone_count].meta['order'] = order[::-1]
                lin = self.read_line(fid)

            elif parts[0]=='limits':
                self.vertices[bone_count].meta['limits'] = [[float(parts[1][1:]),  float(parts[2][:-1])]]

                lin = self.read_line(fid)

                while lin !='end':
                    parts = lin.split()

                    self.vertices[bone_count].meta['limits'].append([float(parts[0][1:]), float(parts[1][:-1])])
                    lin = self.read_line(fid)
                self.vertices[bone_count].meta['limits'] = np.array(self.vertices[bone_count].meta['limits'])

            elif parts[0]=='end':
                self.vertices[bone_count].meta['offset'] = direction*lgth
                lin = self.read_line(fid)

        return lin

    def read_channels(self, fid):
        """Read channels from an acclaim file."""
        bones = [[] for i in self.vertices]
        num_channels = 0
        for vertex in self.vertices:
            num_channels = num_channels + len(vertex.meta['channels'])

        lin = self.read_line(fid)
        while lin != ':DEGREES':
            lin = self.read_line(fid)
            if lin == '':
                raise ValueError('Could not find :DEGREES in ' + fid.name)

        counter = 0
        lin = self.read_line(fid)
        while lin:
            parts = lin.split()
            if len(parts)==1:
                frame_no = int(parts[0])
                if frame_no:
                    counter += 1
                    if counter != frame_no:
                        raise ValueError('Unexpected frame number.')
                else:
                    raise ValueError('Single bone name  ...')
            else:
                ind = self.get_index_by_name(parts[0])
                bones[ind].append(np.array([float(channel) for channel in parts[1:]]))
            lin = self.read_line(fid)

        num_frames = counter

        channels = np.zeros((num_frames, num_channels))

        end_val = 0
        for i in range(len(self.vertices)):
            vertex = self.vertices[i]
            if len(vertex.meta['channels'])>0:                
                start_val = end_val
                end_val = end_val + len(vertex.meta['channels'])
                for j in range(num_frames):
                    channels[j, start_val:end_val] = bones[i][j]
            self.resolve_indices(i, start_val)

        self.smooth_angle_channels(channels)
        return channels

    def writ_channels(self, fid, channels):
        fid.write('#!OML:ASF \n')
        fid.write(':FULLY-SPECIFIED\n')
        fid.write(':DEGREES\n')
        num_frames = channels.shape[0]
        for i_frame in range(num_frames):
            fid.write(str(i_frame+1)+'\n')
            offset = 0
            for vertex in self.vertices:
                fid.write(vertex.name+' '+ ' '.join([str(v) for v in channels[i_frame,offset:offset+len(vertex.meta['channels'])]])+'\n')
                offset += len(vertex.meta['channels'])
        

    def read_documentation(self, fid):
        """Read documentation from an acclaim skeleton file stream."""

        lin = self.read_line(fid)
        while lin[0] != ':':
            self.documentation.append(lin)
            lin = self.read_line(fid)
        return lin

    def read_hierarchy(self, fid):
        """Read hierarchy information from acclaim skeleton file stream."""

        lin = self.read_line(fid)
                    
        while lin != 'end':
            parts = lin.split()
            if lin != 'begin':
                ind = self.get_index_by_name(parts[0])
                for i in range(1, len(parts)):
                    self.vertices[ind].children.append(self.get_index_by_name(parts[i]))
            lin = self.read_line(fid)
        lin = self.read_line(fid)
        return lin

    def read_line(self, fid):
        """Read a line from a file string and check it isn't either empty or commented before returning."""
        lin = '#'
        while lin[0] == '#':
            lin = fid.readline().strip()
            if lin == '':
                return lin
        return lin

    
    def read_root(self, fid):
        """Read the root node from an acclaim skeleton file stream."""
        lin = self.read_line(fid)                    
        while lin[0] != ':':
            parts = lin.split()
            if parts[0]=='order':
                order = []
                for i in range(1, len(parts)):
                    if parts[i].lower()=='rx':
                        chan = 'Xrotation'
                        order.append('x')
                    elif parts[i].lower()=='ry':
                        chan = 'Yrotation'
                        order.append('y')
                    elif parts[i].lower()=='rz':
                        chan = 'Zrotation'
                        order.append('z')
                    elif parts[i].lower()=='tx':
                        chan = 'Xposition'
                    elif parts[i].lower()=='ty':
                        chan = 'Yposition'
                    elif parts[i].lower()=='tz':
                        chan = 'Zposition'
                    elif parts[i].lower()=='l':
                        chan = 'length'
                    self.vertices[0].meta['channels'].append(chan)
                    # order is reversed compared to bvh
                self.vertices[0].meta['order'] = order[::-1]

            elif parts[0]=='axis':
                # order is reversed compared to bvh
                self.vertices[0].meta['axis_order'] = parts[1][::-1].lower()
            elif parts[0]=='position':
                self.vertices[0].meta['offset'] = [float(parts[1]),
                                       float(parts[2]),
                                       float(parts[3])]
            elif parts[0]=='orientation':
                self.vertices[0].meta['orientation'] =  [float(parts[1]),
                                             float(parts[2]),
                                             float(parts[3])]
            lin = self.read_line(fid)
        return lin
    
    def read_skel(self, fid):
        """Loads an acclaim skeleton format from a file stream."""
        lin = self.read_line(fid)
        while lin:
            if lin[0]==':':
                if lin[1:]== 'name':
                    lin = self.read_line(fid)
                    self.name = lin
                elif lin[1:]=='units':
                    lin = self.read_units(fid)
                elif lin[1:]=='documentation':
                    lin = self.read_documentation(fid)
                elif lin[1:]=='root':
                    lin = self.read_root(fid)
                elif lin[1:]=='bonedata':
                    lin = self.read_bonedata(fid)
                elif lin[1:]=='hierarchy':
                    lin = self.read_hierarchy(fid)
                elif lin[1:8]=='version':
                    lin = self.read_line(fid)
                    continue
                else: 
                    if not lin:
                        self.finalize()
                        return
                    lin = self.read_line(fid)
            else:
                raise ValueError('Unrecognised file format')
            self.finalize()
            
    def read_units(self, fid):
        """Read units from an acclaim skeleton file stream."""
        lin = self.read_line(fid)                   
        while lin[0] != ':':
            parts = lin.split()
            if parts[0]=='mass':
                self.mass = float(parts[1])
            elif parts[0]=='length':
                self.length = float(parts[1])
            elif parts[0]=='angle':
                self.angle = parts[1]
            lin = self.read_line(fid)
        return lin

    def resolve_indices(self, index, start_val):
        """Get indices for the skeleton from the channels when loading in channel data."""

        channels = self.vertices[index].meta['channels']
        base_channel = start_val 
        rot_ind = -np.ones(3, dtype=int)
        pos_ind = -np.ones(3, dtype=int)
        for i in range(len(channels)):
            if channels[i]== 'Xrotation':
                rot_ind[0] = base_channel + i
            elif channels[i]=='Yrotation':
                rot_ind[1] = base_channel + i
            elif channels[i]=='Zrotation':
                rot_ind[2] = base_channel + i
            elif channels[i]=='Xposition':
                pos_ind[0] = base_channel + i
            elif channels[i]=='Yposition':
                pos_ind[1] = base_channel + i
            elif channels[i]=='Zposition':
                pos_ind[2] = base_channel + i
        self.vertices[index].meta['rot_ind'] = list(rot_ind)
        self.vertices[index].meta['pos_ind'] = list(pos_ind)

    def set_rotation_matrices(self):
        """Set the meta information at each vertex to contain the correct matrices C and Cinv as prescribed by the rotations and rotation orders."""
        for i in range(len(self.vertices)):
            self.vertices[i].meta['C'] = rotation_matrix(self.vertices[i].meta['axis'][0], 
                                                         self.vertices[i].meta['axis'][1], 
                                                         self.vertices[i].meta['axis'][2], 
                                                         self.vertices[i].meta['axis_order'],
                                                         degrees=True)
            # Todo: invert this by applying angle operations in reverse order
            self.vertices[i].meta['Cinv'] = np.linalg.inv(self.vertices[i].meta['C'])
            

# Utilities for loading in x,y,z data.
def load_text_data(dataset, directory, centre=True):
    """Load in a data set of marker points from the Ohio State University C3D motion capture files (http://accad.osu.edu/research/mocap/mocap_data.htm)."""

    points, point_names = parse_text(os.path.join(directory, dataset + '.txt'))[0:2]
    # Remove markers where there is a NaN
    present_index = [i for i in range(points[0].shape[1]) if not (np.any(np.isnan(points[0][:, i])) or np.any(np.isnan(points[0][:, i])) or np.any(np.isnan(points[0][:, i])))]

    point_names = point_names[present_index]
    for i in range(3):
        points[i] = points[i][:, present_index]
        if centre:
            points[i] = (points[i].T - points[i].mean(axis=1)).T 

    # Concatanate the X, Y and Z markers together
    Y = np.concatenate((points[0], points[1], points[2]), axis=1)
    Y = Y/400.
    connect = read_connections(os.path.join(directory, 'connections.txt'), point_names)
    return Y, connect

def parse_text(file_name):
    """Parse data from Ohio State University text mocap files (http://accad.osu.edu/research/mocap/mocap_data.htm)."""

    # Read the header
    fid = open(file_name, 'r')
    point_names = np.array(fid.readline().split())[2:-1:3]
    fid.close()
    for i in range(len(point_names)):
        point_names[i] = point_names[i][0:-2]

    # Read the matrix data
    S = np.loadtxt(file_name, skiprows=1)
    field = np.uint(S[:, 0])
    times = S[:, 1]
    S = S[:, 2:]

    # Set the -9999.99 markers to be not present
    S[S==-9999.99] = np.NaN

    # Store x, y and z in different arrays
    points = []
    points.append(S[:, 0:-1:3])
    points.append(S[:, 1:-1:3])
    points.append(S[:, 2:-1:3])

    return points, point_names, times

def read_connections(file_name, point_names):
    """Read a file detailing which markers should be connected to which for motion capture data."""

    connections = []
    fid = open(file_name, 'r')
    line=fid.readline()
    while(line):
        connections.append(np.array(line.split(',')))
        connections[-1][0] = connections[-1][0].strip()
        connections[-1][1] = connections[-1][1].strip()
        line = fid.readline()
    connect = np.zeros((len(point_names), len(point_names)),dtype=bool)
    for i in range(len(point_names)):
        for j in range(len(point_names)):
            for k in range(len(connections)):
                if connections[k][0] == point_names[i] and connections[k][1] == point_names[j]:
                    
                    connect[i,j]=True
                    connect[j,i]=True
                    break
    
    return connect

    
  
skel = acclaim_skeleton()



