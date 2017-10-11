#!/usr/bin/python
#
# python_render_ex4

import sys, os, random
from math import *
from cgkit.cgtypes import *
from cgkit.cmds import *
from cgkit.pluginmanager import *
from cgkit.riutil import *
import ode


def draw_body(body):
    """
    Draw the body with Renderman primitives.
    """
    (x, y, z) = body.getPosition()
    R = body.getRotation()
    # Construct the transformation matrix
    T = mat4()
    T.setMat3(mat3(R))
    T[3] = ( x, y, z, 1.0 )
    RiTransformBegin()
    # Apply rotation & translation
    RiTransform(T.toList())
    # Draw the body
    RiAttributeBegin()
    RiColor([ 1.0, 0.6, 0.0 ])
    # Create the polygons
    for i in range(len(mesh.faces)):
        RiPolygon(P = [ [ mesh.verts[mesh.faces[i][j]][k] for k in range(3) ] for j in range(3) ])
    RiAttributeEnd()
    RiTransformEnd()

def create_body(density, r, h):
    """
    Create a cylindrical body and its corresponding geom.
    """
    # Create body
    body = ode.Body(world)
    M = ode.Mass()
    M.setCylinder(density, 2, r, h)
    body.setMass(M)
    # Create a trimesh geom for collision detection
    tm = ode.TriMeshData()
    tm.build(mesh.verts, mesh.faces)
    geom = ode.GeomTriMesh(tm, space)
    geom.setBody(body)
    return(body, geom)

# drop_object
def drop_object():
    """
    Drop an object into the space.
    """
    global bodies, geom
    # Create the body
    (body, geom) = create_body(1000, 0.55, 0.80)
    # Set a position
    body.setPosition( (0.0, 5.0, 0.0) )
    # Set a random rotation
    m = mat3().rotation(objcount*pi/2 + random.gauss(-pi/4, pi/4), (0, 1, 0))
    body.setRotation(m.toList())
    # Append the body and geom to a list
    bodies.append(body)
    geoms.append(geom)

# Collision callback
def near_callback(args, geom1, geom2):
    """
    Callback function for the collide method. This function checks if the given
    geoms do collide and creates contact joints if they do.
    """
    # Check if the objects do collide
    contacts = ode.collide(geom1, geom2)
    # Create contact joints
    (world, contactgroup) = args
    for c in contacts:
        c.setBounce(0.2)
        c.setMu(5000)
        j = ode.ContactJoint(world, contactgroup, c)
        j.attach(geom1.getBody(), geom2.getBody())


################################################################################

# Set the output rib file
renderer = "pyode_ex4.rib"

# Create the header
RiBegin(renderer)
RiuDefaultHeader()

# Set the width & height for each frame
#width = 640
#height = 480
width = 1280
height = 960

# Set the simulation time, number of frames, iterations/frames ratio and number of iterations
T = 20.0
NFRAMES = 501
R = 20
N = (NFRAMES - 1)*R + 1

#N = 9001
#NFRAMES = 5
#R = (N - 1)/(NFRAMES - 1)

# Set the frame rate and iteration time step
fps = float(NFRAMES - 1)/T # 25
dt = T/(N - 1) #ips = 1.0/dt

# Specify the object drop time & count
objdroptime = 0.6
objdropcount = int(objdroptime/dt)
numobjects = 25

# Initialize the random number generator
random.seed(0)

# Create a world object
world = ode.World()
world.setGravity( (0, -9.81, 0) )
world.setERP(0.8)
world.setCFM(1E-5)

# Create a space object
space = ode.Space()

# Create a plane geom which prevent the objects from falling forever
floor = ode.GeomPlane(space, (0, 1, 0), 0)

# A list with ODE bodies
bodies = []

# The geoms for each of the bodies
geoms = []

# A joint group for the contact joints that are generated whenever two bodies collide
contactgroup = ode.JointGroup()

# Create a plugin manager to facilitate obj import
pm = PluginManager()
pdesc = pm.importPlugin("myobjimport.py")
if pdesc.status!=STATUS_OK:
    print "Error: Unable to import Plugin!"
# Search for the plugin class
objdesc = pm.findObject("myobjimport.MyOBJImporter")
# Create an instance of the plugin class
PluginClass = objdesc.object
objreader = PluginClass()

# Import the obj
objreader.importFile("teapot.obj")

#listWorld()

# Access the mesh data
mesh = worldObject("Mesh")

# Initialize the simulation
frame = 0
state = 0
counter = 0
objcount = 0

# Run the simulation
for n in range(N):
    # Detect collisions and create contact joints
    space.collide((world, contactgroup), near_callback)

    # Simulation step
    world.step(dt)

    # Remove all contact joints
    contactgroup.empty()

    counter += 1
    if (state == 0):
        if (counter == objdropcount):
            # Drop an object
            drop_object()
            objcount += 1
            counter = 0
        if objcount == numobjects:
            state = 1
            counter = 0

    if (n % R == 0):
        # Create a frame
        frame += 1
#        print "Frame: %d" % frame
        filename = "frames/pyode_ex4.%03d.tif" % frame

        RiFrameBegin(frame)
        # Set the projection
        RiProjection(RI_PERSPECTIVE, fov=22)

        # Create a view transformation and apply
        V = mat4(1.0).lookAt(4.0*vec3(-2.0, 3.0, -4.0), (0, 0.5, 0), up = (0, 1, 0))
        V = V.inverse()
        RiTransform(V.toList())

        # Set the output file and frame format
        RiDisplay(filename, RI_FILE, RI_RGBA)
        RiFormat(width, height, 1.0)

        # Apply sampling
        RiPixelSamples(4,4)
        RiShadingInterpolation(RI_SMOOTH)

        # Begin the world    
        RiWorldBegin()
        # Make objects visible to rays
        RiDeclare("diffuse", "integer")
        RiAttribute("visibility", "diffuse", 1)
        RiDeclare("bias", "float")
        RiAttribute("trace", "bias", 0.005)
        # Don't interpolate irradiance
        RiDeclare("maxerror", "float")
        RiAttribute("irradiance", "maxerror", 0.0)
        # Apply global illumination
        RiDeclare("samples", "integer")
        RiSurface("occsurf2", "samples", 256)

        # Create a white ground plane
        RiAttributeBegin()
        RiColor([ 0.9, 0.9, 0.9 ])
        RiPolygon(P = [ -20, 0, 20, 20, 0, 20, 20, 0, -20, -20, 0, -20 ])
        RiAttributeEnd()

        # Draw the bodies
        for b in bodies:
            draw_body(b)

        # Complete the world & frame
        RiWorldEnd()
        RiFrameEnd()

# Create the termination line
RiEnd()
