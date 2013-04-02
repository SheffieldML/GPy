import os
import numpy as np

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

    
  
