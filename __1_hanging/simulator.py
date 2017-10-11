# run simulation.


from mcode import *
import sys, os, random, time
from math import *
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
import ode
import Ragdoll_with_hanger


def main():
    # initialize GLUT
    glutInit([])
    glutInitDisplayMode(GLUT_RGB | GLUT_DEPTH | GLUT_DOUBLE)
    

    # create the program window
    x = 0
    y = 0
    width = 640
    height = 480
    glutInitWindowPosition(x, y);
    glutInitWindowSize(width, height);
    glutCreateWindow("PyODE Ragdoll Simulation")
    

    # create an ODE world object
    global world
    world = ode.World()
    world.setGravity((0.0, -9.81, 0.0))
    world.setERP(0.1)
    world.setCFM(1E-4)

    # create an ODE space object
    global space
    space = ode.Space()

    # create a plane geom to simulate a floor
    floor = ode.GeomPlane(space, (0, 1, 0), 0)

    # create a list to store any ODE bodies which are not part of the ragdoll (this
    #   is needed to avoid Python garbage collecting these bodies)
    global bodies
    bodies = []

    # create a joint group for the contact joints generated during collisions
    #   between two bodies collide
    global contactgroup
    contactgroup = ode.JointGroup()

    # set the initial simulation loop parameters
    global Paused, lasttime, numiter, fps, dt, SloMo, stepsPerFrame
    fps = 60
    dt = 1.0 / fps
    stepsPerFrame = 2
    SloMo = 1
    Paused = False
    lasttime = time.time()
    numiter = 0


    # create the ragdoll
    global ragdoll
    #ragdoll = Ragdoll.Ragdoll(world, space, 500, (0.0, 1.9, 0.0))
    ragdoll = Ragdoll_with_hanger.Ragdoll_with_hanger(world, space, 500, (0.0, 0.0, 0.0))
    print "total mass is %.1f kg (%.1f lbs)" % (ragdoll.totalMass, ragdoll.totalMass * 2.2)

    # set GLUT callbacks
    glutKeyboardFunc(onKey)
    glutDisplayFunc(onDraw)
    glutIdleFunc(onIdle)

    # enter the GLUT event loop
    glutMainLoop()

def prepare_GL():
    """Setup basic OpenGL rendering with smooth shading and a single light."""

    glClearColor(0.8, 0.8, 0.9, 0.0)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_LIGHTING)
    glEnable(GL_NORMALIZE)
    glShadeModel(GL_SMOOTH)

    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective (45.0, 1.3333, 0.2, 20.0)

    glViewport(0, 0, 640, 480)

    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()

    glLightfv(GL_LIGHT0,GL_POSITION,[0, 0, 1, 0])
    glLightfv(GL_LIGHT0,GL_DIFFUSE,[1, 1, 1, 1])
    glLightfv(GL_LIGHT0,GL_SPECULAR,[1, 1, 1, 1])
    glEnable(GL_LIGHT0)

    glEnable(GL_COLOR_MATERIAL)
    glColor3f(0.8, 0.8, 0.8)

    gluLookAt(1.5, 4.0, 3.0, 0.5, 1.0, 0.0, 0.0, 1.0, 0.0)



# polygon resolution for capsule bodies
CAPSULE_SLICES = 16
CAPSULE_STACKS = 12

def draw_body(body):
    """Draw an ODE body."""
    rot = makeOpenGLMatrix(body.getRotation(), body.getPosition())
    glPushMatrix()
    glMultMatrixd(rot)
    if body.shape == "capsule":
        cylHalfHeight = body.length / 2.0
        glBegin(GL_QUAD_STRIP)
        for i in range(0, CAPSULE_SLICES + 1):
            angle = i / float(CAPSULE_SLICES) * 2.0 * pi
            ca = cos(angle)
            sa = sin(angle)
            glNormal3f(ca, sa, 0)
            glVertex3f(body.radius * ca, body.radius * sa, cylHalfHeight)
            glVertex3f(body.radius * ca, body.radius * sa, -cylHalfHeight)
        glEnd()
        glTranslated(0, 0, cylHalfHeight)
        glutSolidSphere(body.radius, CAPSULE_SLICES, CAPSULE_STACKS)
        glTranslated(0, 0, -2.0 * cylHalfHeight)
        glutSolidSphere(body.radius, CAPSULE_SLICES, CAPSULE_STACKS)
    elif body.shape == "box":
        sx, sy, sz = body.radius, body.radius, body.length
        glScalef(sx, sy, sz)
        glutSolidCube(1)
    glPopMatrix()
    


def onKey(c, x, y):
    """GLUT keyboard callback."""

    global SloMo, Paused

    # set simulation speed
    if c >= '0' and c <= '9':
        SloMo = 4 * int(c) + 1
    # pause/unpause simulation
    elif c == 'p' or c == 'P':
        Paused = not Paused
    # quit
    elif c == 'q' or c == 'Q':
        sys.exit(0)

    elif c == 'h' or c == 'H':
        explosion('head')
        print 'impact on head !!'
    elif c == 't' or c == 'T':
        applyTorque('Hip', (300, 0, 0))
        print 'apply torque at Hip !!'
        


def onDraw():
    """GLUT render callback."""

    prepare_GL()

    for b in bodies:
        draw_body(b)
    for b in ragdoll.bodies:
        draw_body(b)

    glutSwapBuffers()


def onIdle():
    """GLUT idle processing callback, performs ODE simulation step."""

    if Paused:
        return

    global lasttime
    t = dt - (time.time() - lasttime)
    if (t > 0):
        time.sleep(t)


    glutPostRedisplay()

    for i in range(stepsPerFrame):
        # Detect collisions and create contact joints
        space.collide((world, contactgroup), near_callback)

        # Simulation step (with slo motion)
        world.step(dt / stepsPerFrame / SloMo)

        global numiter
        numiter += 1

        # apply internal ragdoll forces
        ragdoll.update()

        # Remove all contact joints
        contactgroup.empty()

    lasttime = time.time()


def near_callback(args, geom1, geom2):
	"""
	Callback function for the collide() method.

	This function checks if the given geoms do collide and creates contact
	joints if they do.
	"""

	if (ode.areConnected(geom1.getBody(), geom2.getBody())):
		return

	# check if the objects collide
	contacts = ode.collide(geom1, geom2)

	# create contact joints
	world, contactgroup = args
	for c in contacts:
		c.setBounce(0.2)
		c.setMu(500) # 0-5 = very slippery, 50-500 = normal, 5000 = very sticky
		j = ode.ContactJoint(world, contactgroup, c)
		j.attach(geom1.getBody(), geom2.getBody())

def explosion(target):
    """ simulate explosion.
    The force is dependant on the object distance from the origin.
    """

    global ragdoll
    
    if target == 'head':
        b = ragdoll.doll.head
        l = b.getPosition()
        d = length(l)
        a = max(0, 40000*(1.0-0.2*d*d))
        l = [l[0]/4, l[1], l[2]/4]
        scalp(l, a/d)
        b.addForce(l)
    
    """
    if target == 'head':
        b = ragdoll.head
        b.setPosition((random.gauss(0,0.1), 1.0, random.gauss(0,0.1)))
    """
def applyTorque(target, val):
    if target == 'Hip':
        b1 = ragdoll.doll.rightUpperLeg
        b2 = ragdoll.doll.leftUpperLeg
        b1.addRelTorque(val)
        b2.addRelTorque(val)
        



if __name__ == "__main__":
    main()









