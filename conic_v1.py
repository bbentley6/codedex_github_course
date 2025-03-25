import numpy as np
import CGNS.MAP as CGM
import CGNS.PAT.cgnslib as CGL
import CGNS.PAT.cgnskeywords as CK
import sys
from scipy.optimize import minimize_scalar
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# from pyhyp import pyHyp


# import CGNS.PAT.cgnsutils as CU
# import CGNS.VAL.simplecheck as CGV # ModuleNotFoundError: No module named 'imp'

# HERE ARE THE FUNCTION CALLS
# https://pycgns.github.io/PAT/lib.html#CGNS.PAT.cgnslib.newZone
# 
# FROM THE USER MANUAL IN FORTRAN, UNDER Single-zone structured grid 
# https://cgns.github.io/doc/userguide.html#docuserguide

# Coordinate systems: https://cgns.github.io/standard/SIDS/appendix_a.html#a1-coordinate-systems
# Cartesian Coordinate System: CoordinateX, CoordinateY, CoordinateZ
# Cylindrical Coordinate System: CoordinateR, CoordinateTheta, CoordinateZ

# Define classes
class Inputs:
    sphericalBluntingRadius = 0
    conicLengths = 0
    conicHalfAnglesDeg = 0
    transitionSpacing = 0
    nnodes = 0
    conicHalfAnglesRad = np.deg2rad(conicHalfAnglesDeg)
    theta_nnodes = 0

class Vertices:
    def __init__(self, x=0, y=0, z=0):
        self.x = x
        self.y = y
        self.z = z

def read_file() -> Inputs:
    ##################################
    ## hard-code inputs for example ##
    ##################################
    inputs = Inputs()
    inputs.sphericalBluntingRadius = 0.5
    inputs.conicLengths    = np.array([5.,10.,20.,5.])
    inputs.conicHalfAnglesDeg = np.array([5.,15.,0.,-20.])
    inputs.transitionSpacing = np.array([0.025, 0.04, 0.2, 0.2, 0.2, 0.2])
    inputs.nnodes = np.array([15, 25, 20, 20, 10], dtype=int)
    inputs.theta_nnodes = 100
    inputs.conicHalfAnglesRad = np.deg2rad(inputs.conicHalfAnglesDeg)
    return inputs

def rect_volume_mesh():
    # create a structured rectangular grid to start off
    n_cells_x = 20
    n_cells_y = 16
    n_cells_z = 8

    # the number of vertices is one more than the number of cells in each direction
    n_vertices_x = n_cells_x + 1
    n_vertices_y = n_cells_y + 1
    n_vertices_z = n_cells_z + 1

    length_x = n_cells_x
    length_y = n_cells_y
    length_z = n_cells_z

    # Creating 3D arrays for x, y, and z coordinates
    x = np.zeros((n_vertices_x, n_vertices_y, n_vertices_z))
    y = np.zeros((n_vertices_x, n_vertices_y, n_vertices_z))
    z = np.zeros((n_vertices_x, n_vertices_y, n_vertices_z))

    # Iterating over the cube to assign coordinates
    for k in range(n_vertices_z):
        for j in range(n_vertices_y):
            for i in range(n_vertices_x):
                x[i, j, k] = i
                y[i, j, k] = j
                z[i, j, k] = k

    # x_points = np.linspace(0, length_x, n_vertices_x)
    # y_points = np.linspace(0, length_y, n_vertices_y)
    # z_points = np.linspace(0, length_z, n_vertices_z)

    # x_mesh, y_mesh, z_mesh = np.meshgrid(x_points, y_points, z_points, indexing='ij')
    # points = np.vstack([x_mesh.ravel(), y_mesh.ravel(), z_mesh.ravel()]).T

    # # declare arrays
    # x_array = []
    # y_array = []
    # z_array = []

    # # cycle through points to get the x, y, and z values
    # for point in points:
    #     x_array.append(point[0])
    #     y_array.append(point[1])
    #     z_array.append(point[2])

def multiconic_surface_mesh(inputs: Inputs) -> Vertices:
    # init the nodes and node normals (2D) arrays
    endIndex = np.cumsum(inputs.nnodes-1, dtype=int)
    startIndex = np.empty_like(endIndex)
    startIndex[0]  = 0
    startIndex[1:] = endIndex[:-1]
    nodes2D   = np.zeros((inputs.nnodes.sum()-inputs.nnodes.size+1, 2), dtype=np.double)
    normals2D = np.zeros_like(nodes2D)
    
    #####################################################
    ##         the spherical blunting segment          ##
    #####################################################
    # determine the blunting wetted length
    bluntingArc = (np.pi/2.)-inputs.conicHalfAnglesRad[0] # angle of hemisphere
    bluntingWL = inputs.sphericalBluntingRadius * bluntingArc

    # determine the blunting distribution in terms of each node's arc
    eta = np.linspace(0., 1., inputs.nnodes[0])
    def findDelta(delta):
        u = 0.5 * (1. + np.tanh(delta * (eta-0.5))/np.tanh(delta/2.))
        return np.abs(u[1]*bluntingWL-inputs.transitionSpacing[0])
    res = minimize_scalar(findDelta, bounds=[0., 100.])
    if res:
        delta =  res.x
        if delta>99.:
            print('WARNING - unable to exactly match specified tanh distribution.')
    else:
        print('failed to find tanh distribution for specififed parameters')
        sys.exit()
    u = 0.5 * (1. + np.tanh(delta * (eta-0.5))/np.tanh(delta/2.))
    A = np.sqrt(inputs.transitionSpacing[1]/bluntingWL)/np.sqrt(inputs.transitionSpacing[0]/bluntingWL)
    s = u / (A+(1-A)*u)
    s *= bluntingWL
    bluntingNodalArcs = s / inputs.sphericalBluntingRadius

    # convert blunting arc to XY coordinates
    nodes2D[:inputs.nnodes[0],0] = inputs.sphericalBluntingRadius - (np.cos(bluntingNodalArcs) * inputs.sphericalBluntingRadius)
    nodes2D[:inputs.nnodes[0],1] =                                  (np.sin(bluntingNodalArcs) * inputs.sphericalBluntingRadius)

    # convert blunting arc into normal vector (normalize it later)
    normals2D[:inputs.nnodes[0],0] = nodes2D[:inputs.nnodes[0],0] - inputs.sphericalBluntingRadius
    normals2D[:inputs.nnodes[0],1] = nodes2D[:inputs.nnodes[0],1]

    #####################################################
    ##               the conic segments                ##
    #####################################################
    for conicLength, conicHalfAngleRad, distStart, distEnd, nn, i in zip(inputs.conicLengths, inputs.conicHalfAnglesRad, inputs.transitionSpacing[1:], inputs.transitionSpacing[2:], inputs.nnodes[1:], startIndex[1:]):
        startingCoordinate = nodes2D[i]
        endingCoordinate = np.array([startingCoordinate[0]+conicLength,
                                     startingCoordinate[1]+conicLength*np.tan(conicHalfAngleRad)])
        edgeVector = endingCoordinate - startingCoordinate
        wl = np.linalg.norm(edgeVector)

        # determine the tanh distribution of node coordinates
        eta = np.linspace(0., 1., nn)
        A = np.sqrt(distEnd)/np.sqrt(distStart)
        B = 1. / (wl * np.sqrt(distEnd*distStart))
        findDelta = lambda delta : np.abs(B - np.sinh(delta)/delta)
        res = minimize_scalar(findDelta, bounds=[0., 100.])
        if res:
            delta =  res.x
            if delta>99.:
                print('WARNING - unable to exactly match specified tanh distribution.')
        else:
            print('failed to find tanh distribution for specififed parameters')
            sys.exit()
        u = 0.5 * (1. + np.tanh(delta * (eta/wl-0.5))/np.tanh(delta/2.))
        s = u / (A+(1-A)*u)
        s *= wl
        nodes2D[i:i+nn] = np.einsum('...i,j', s, edgeVector) + startingCoordinate

        # compute the node normals (all the same, except the first, which averages with previous segment)
        edgeVector /= wl # make a unit vector

    x_nodes2D = nodes2D[:,0]
    y_nodes2D = nodes2D[:,1]

    plt.plot(x_nodes2D, y_nodes2D)
    plt.quiver(x_nodes2D, y_nodes2D, normals2D[:,0], normals2D[:,1])
    plt.axis('scaled')
    mpl.use('TkAgg')
    plt.show()

    # convert from xy in 2D to r,theta,z in 3D. Theta is in radians.
    # x -> z, y -> r, theta created
    r_nodes2D = y_nodes2D
    theta_nodes2D = np.linspace(0, 2*np.pi*(inputs.theta_nnodes-1)/(inputs.theta_nnodes), inputs.theta_nnodes)
    z_nodes2D = x_nodes2D

    # Creating 3D arrays for r, theta, and z coordinates. There is only 1 r-value for each z-location.
    r_nodes3D_cyl = np.zeros((z_nodes2D.size, theta_nodes2D.size, 1))
    theta_nodes3D_cyl = np.empty_like(r_nodes3D_cyl)
    z_nodes3D_cyl = np.empty_like(r_nodes3D_cyl)
    x_nodes3D_rect = np.empty_like(r_nodes3D_cyl)
    y_nodes3D_rect = np.empty_like(r_nodes3D_cyl)
    z_nodes3D_rect = np.empty_like(r_nodes3D_cyl)

    # Iterating to assign coordinates
    for j in range(theta_nodes2D.size):
        for i in range(z_nodes2D.size):
            r_nodes3D_cyl[i, j, 0] = r_nodes2D[i]
            theta_nodes3D_cyl[i, j, 0] = theta_nodes2D[j]
            z_nodes3D_cyl[i, j, 0] = z_nodes2D[i]
            x_nodes3D_rect[i, j, 0] = z_nodes3D_cyl[i, j, 0]
            y_nodes3D_rect[i, j, 0] = r_nodes3D_cyl[i, j, 0] * np.sin(theta_nodes3D_cyl[i, j, 0])
            z_nodes3D_rect[i, j, 0] = r_nodes3D_cyl[i, j, 0] * np.cos(theta_nodes3D_cyl[i, j, 0])        

    # Create a 3D plot
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot
    ax.scatter(x_nodes3D_rect, y_nodes3D_rect, z_nodes3D_rect, c=z_nodes3D_rect, cmap='viridis', marker='o')

    # Labels
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')
    ax.set_title('3D Scatter Plot')

    # Ensure the axes have the same scale
    x_min, x_max = np.min(x_nodes3D_rect), np.max(x_nodes3D_rect)
    y_min, y_max = np.min(y_nodes3D_rect), np.max(y_nodes3D_rect)
    z_min, z_max = np.min(z_nodes3D_rect), np.max(z_nodes3D_rect)

    # Find the max range
    max_range = max(x_max - x_min, y_max - y_min, z_max - z_min) / 2.0

    # Compute midpoints
    x_mid = (x_max + x_min) / 2.0
    y_mid = (y_max + y_min) / 2.0
    z_mid = (z_max + z_min) / 2.0

    # Set equal aspect ratio
    ax.set_xlim(x_mid - max_range, x_mid + max_range)
    ax.set_ylim(y_mid - max_range, y_mid + max_range)
    ax.set_zlim(z_mid - max_range, z_mid + max_range)

    # Show plot
    plt.show()

    verticesSurface = Vertices(x_nodes3D_rect, y_nodes3D_rect, z_nodes3D_rect)
    return verticesSurface

def write_plot3D_ascii(vertices: Vertices, filename: str) -> str:
    """Writes grid data to an ASCII Plot3D file, for a single block.

    Args:
        filename (str): Name of the output file.
        vertices (Vertices): The x-, y-, and z-values of each vertex.
    """
    
    with open(filename, 'w') as f:
        # Number of blocks
        f.write(f"1\n")

        X = vertices.x
        Y = vertices.y
        Z = vertices.z

        # Write grid dimensions for the block
        IMAX, JMAX, KMAX = X.shape
        f.write(f"{IMAX} {JMAX} {KMAX}\n")

        # Write coordinate values
        for array in [X, Y, Z]:  # Write X, then Y, then Z
            for k in range(array.shape[2]):
                for j in range(array.shape[1]):
                    for i in range(array.shape[0]):
                        f.write(f"{array[i, j, k]:.6f} ")
        f.write("\n")  # Newline after each X, Y, or Z array

    return filename

def output_to_cgns_2D(verticesSurface: Vertices) -> str:
    # Create CGNS file
    filename = 'rectangle_simple_mesh.cgns'
    
    # Create CGNS Tree
    tree = CGL.newCGNSTree()

    # Create CGNS Base
    base = CGL.newCGNSBase(tree, 'Base1', 3, 3) # tree, name, cell dimension (2 for a surface/face cell, 3 for volume), physical dimension
    # zsize = np.array([[n_cells_x],[n_cells_y],[n_cells_z]], dtype=np.int32)

    # declare the vectors
    n_vertices = np.array([n_vertices_x, n_vertices_y, n_vertices_z])
    n_cells = np.array([n_cells_x, n_cells_y, n_cells_z])
    size_boundary_vertex = np.array([0,0,0]) # always 0's for structured grids

    # Create a Fortran-contiguous array, required otherwise pyCGNS throws an error
    zsize = np.array([n_vertices, n_cells, size_boundary_vertex]).T
    # zsize = np.asfortranarray(zsize)
    zsize = np.array(zsize, order="F") # "F" is column-major (Fortran-style) order.

    # combine the vectors into a matrix
    # isize = np.vstack((n_vertices.reshape(-1,1), n_cells.reshape(-1,1), size_boundary_vertex.reshape(-1,1)))

    # # print values
    # print("zsize: ")
    # print(zsize)
    # print("x.shape: ", x.shape)
    # print("y.shape: ", y.shape)
    # print("z.shape: ", z.shape)

    # create a zone
    # s=np.array([[n_vertices_x],[n_vertices_y],[n_vertices_z]],dtype=np.int32)
    # zone = CGL.newZone(base, 'Zone1', s, CK.Structured_s)
    zone = CGL.newZone(base, 'Zone1', zsize, CK.Structured_s)
    # check_value = cgu.checkNodeCompliant(zone)

    # Add Grid Coordinates to the zone
    gc = CGL.newGridCoordinates(zone, CK.GridCoordinates_s)
    CGL.newDataArray(gc, CK.CoordinateX_s, x)
    CGL.newDataArray(gc, CK.CoordinateY_s, y)
    CGL.newDataArray(gc, CK.CoordinateZ_s, z)

    # grid coordinates added as a single vector (gives error "Wrong number of dimension in DataArray CoordinateX" in Tecplot)
    # CGL.newDataArray(gc, CK.CoordinateX_s, x_array)
    # CGL.newDataArray(gc, CK.CoordinateY_s, y_array)
    # CGL.newDataArray(gc, CK.CoordinateZ_s, z_array)

    # save as a cgns file
    CGM.save(filename, tree)

def output_to_cgns_3D(verticesVol: Vertices):
    print("output")

# define the main function
def main():
    filenameSurface = "temp.xyz"
    
    inputs = read_file() # read in the batch file
    vertices = multiconic_surface_mesh(inputs) # create the vertices of the 2D surface mesh
    write_plot3D_ascii(vertices, filenameSurface) # create a surface mesh cgns file
    # write_plot3D_ascii(vertices, filenameVolume) # create a volume mesh cgns file

# Run the main function when the script is executed directly
if __name__ == "__main__":
    main()