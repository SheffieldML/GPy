import os
import numpy as np

def load_text_data(dataset, directory, centre=True):
    """Load in a data set of marker points from the Ohio State University C3D motion capture files (http://accad.osu.edu/research/mocap/mocap_data.htm)."""

    points, point_names = parse_text(os.path.join(directory, dataset + '.txt'))[0:2]
    # Remove markers where there is a NaN
    present_index = [i for i in range(points[0].shape[1]) if not (np.any(np.isnan(points[0][:, i])) or np.any(np.isnan(points[0][:, i])) or np.any(np.isnan(points[0][:, i])))]

    # Concatanate the X, Y and Z markers together
    Y = np.concatenate((points[0][:, present_index], points[1][:, present_index], points[2][:, present_index]), axis=1)
    if centre:
        Y = Y - Y.mean(axis=0)
    Y = Y/400.
    return Y

def parse_text(file_name):
    """Parse data from Ohio State University text mocap files (http://accad.osu.edu/research/mocap/mocap_data.htm)."""

    # Read the header
    fid = open(file_name, 'r')
    point_names = np.array(fid.readline().split())[2:-1:3]
    fid.close()
    for i in range(len(point_names)):
        point_names[i] = point_names[i][0:-3]

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

#def read_connections():


#     fid = fopen(fileName);
#     i = 1;
#     rem = fgets(fid);	
#     while(rem ~= -1)		
#         [token, rem] = strtok(rem, ',');
#         connections{i, 1} = fliplr(deblank(fliplr(deblank(token))));
#         [token, rem] = strtok(rem, ',');
#         connections{i, 2} = fliplr(deblank(fliplr(deblank(token))));
#         i = i + 1;
#         rem = fgets(fid);	
#     end

#     connect = zeros(length(pointNames));
#     fclose(fid);
#     for i = 1:size(connections, 1);
#         for j = 1:length(pointNames)
#             if strcmp(pointNames{j}, connections{i, 1}) | ...
#                 strcmp(pointNames{j}, connections{i, 2})
#       for k = 1:length(pointNames)
#         if k == j
#           break
#         end
#         if strcmp(pointNames{k}, connections{i, 1}) | ...
#               strcmp(pointNames{k}, connections{i, 2})
#           connect(j, k) = 1;
#         end
#       end
#     end
#   end
# end
# connect = sparse(connect);      
