# run simulation.


from mcode import *
import sys, os, random, time
from math import *
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
import ode
import InvertedPendulum as pd
from PIL import Image


# parameters for control motor.
TORQUE = 80
ROTATE_OFFSET = 8
MAXTORQUE = 200
global TARGET_ANGLE


# paremeters for reinforcement learning policy.
EPS_ANG_THRES = 31.5


class Simulator():
    
    def __init__(self):
        # initialize GLUT
        glutInit([])
        glutInitDisplayMode(GLUT_RGB | GLUT_DEPTH | GLUT_DOUBLE)
    

        # create the program window
        x = 0
        y = 0
        global width, height
        width = 640
        height = 480
        glutInitWindowPosition(x, y);
        glutInitWindowSize(width, height);
        self.win = glutCreateWindow("PyODE Inverted Pendulum Simulation")
        self.reset()

    def reset(self):
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
        stepsPerFrame = 1
        SloMo = 1
        Paused = False
        lasttime = time.time()
        numiter = 0

        # reward.
        global reward
        reward = 0


        # create the pendulum
        global pendulum
        pendulum = pd.Pendulum(world, space, 500, (0.5,0.0,0.3))
        print "total mass is %.1f kg (%.1f lbs)" % (pendulum.totalMass, pendulum.totalMass * 2.2)

        global TARGET_ANGLE
        TARGET_ANGLE = pendulum.hHinge.getAngle()

        # set GLUT callbacks
        #glutKeyboardFunc(self.onKey)
        #glutDisplayFunc(onDraw)
        #glutIdleFunc(onIdle)

        # enter the GLUT event loop
        self.SpinOnce()
        #glutMainLoop()

    def DestroyWindow(self):
        glutDestroyWindow(self.win)

    def prepare_GL(self):
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




    def draw_body(self, body):
        # polygon resolution for capsule bodies
        CAPSULE_SLICES = 16
        CAPSULE_STACKS = 12
        """Draw an ODE body."""
        rot = makeOpenGLMatrix(body.getRotation(), body.getPosition())
        glPushMatrix()
        glMultMatrixd(rot)
        if body.shape == "cylinder":
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
        elif body.shape == "box":
            sx, sy, sz = body.height, body.width, body.len
            glScalef(sx, sy, sz)
            glutSolidCube(1)
        glPopMatrix()
        


    def onKey(self, c, x, y):
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

        elif c == 'z' or c == 'Z':
            applyTorque('hbar', (300, 0, 0))
            print 'apply counter-clw torque at HBar !!'
        
        elif c == 'x' or c == 'X':
            applyTorque('hbar', (-300, 0, 0))
            print 'apply clw torque at HBar !!'
        
        elif c =='s' or c == 'S':
            PILImage = snapshot(width, height)
            dummy = preprocess(PILImage)
            print 'snapshot!!'
            
        elif c =='i' or c == 'I':
            #applyTorque(TORQUE)
            self.rotate(ROTATE_OFFSET)
            self.SpinOnce()
            print 'rotate in cw direction'

        elif c =='o' or c == 'O':
            #applyTorque(-TORQUE)
            self.rotate(-ROTATE_OFFSET)
            self.SpinOnce()
            print 'rotate in ccw direction'
        
        elif c =='n' or c == 'N':
            SpinOnce()
            onDraw()
            ang = get_state()
            print ang
            print '1 step forward.'



    def onDraw(self):
        """GLUT render callback."""

        self.prepare_GL()

        for b in bodies:
            self.draw_body(b)
        for b in pendulum.bodies:
            self.draw_body(b)

        glutSwapBuffers()


    def onIdle(self):
        """GLUT idle processing callback, performs ODE simulation step."""

        if Paused:
            return

        global lasttime
        t = dt - (time.time() - lasttime)
        if (t > 0):
            time.sleep(t)


        glutPostRedisplay()


        # forward DQN and decide action.
        #print 'dqn.'
        
        # PD Control.
        self.ControlLoop()
        
        # simulate for 'stepsPerFrame' steps.
        self.SpinOnce()


        lasttime = time.time()



    # process for one step.
    def SpinOnce(self):
        for i in range(stepsPerFrame):
            # PD Control
            self.ControlLoop()

            # Detect collisions and create contact joints
            space.collide((world, contactgroup), self.near_callback)

            # Simulation step (with slo motion)
            world.step(dt / stepsPerFrame / SloMo)

            global numiter
            numiter += 1

            # apply internal pendulum forces
            pendulum.update()

            # Remove all contact joints
            contactgroup.empty()

            # draw
            self.onDraw()

    def get_vertical_angle(self):
        b = pendulum.vHinge
        radian = b.getAngle()
        ang = radian/3.141592*180
        return ang

    def step(self, action):
        # if action == pos --> apply torque.
        if action == 0:
            self.rotate(ROTATE_OFFSET)
        elif action == 1:
            self.rotate(-ROTATE_OFFSET)
        
        curAng = self.get_vertical_angle()
        
        # check if vertical bar is too far from the origin axis
        done = False
        if abs(curAng) > EPS_ANG_THRES:
            done = True
        
        # return reward.
        global reward
        reward += 1
        
        self.SpinOnce()
        
        return reward, done


    def near_callback(self, args, geom1, geom2):
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


    def snapshot(self):
        buffer = glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE)
        image = Image.frombytes(mode="RGB", size=(width, height), 
                                 data=buffer)
        image = image.transpose(Image.FLIP_TOP_BOTTOM)
        return image

    # rotate hbar in clock-wise direction.
    def applyTorque(self, val):
        b = pendulum.hHinge
        b.addTorque(val)


    # Posotion control.
    def PD_controller(self, targAng):
        Kp = 400
        Kd = 40              ## should be 2*sqrt(Kp)
        PI = 3.141592
        targAng = targAng/180.0*PI
        #print targAng
        b = pendulum.hHinge
        curAng = b.getAngle()
        curVel = b.getAngleRate()
        

        while(targAng > 2*PI):
            targAng = targAng - 2*PI
        while(targAng < 0):
            targAng = targAng + 2*PI
            #if targAng == PI:
            #    targAng = targAng = targANg-2*PI

        if curAng < 0:
            curAng = 2*PI + curAng
        if targAng < 0:
            targAng = 2*PI + targAng
        
        diffPos = curAng - targAng
        if diffPos > 0.75*PI:
            targAng = targAng + 2*PI
            diffPos = curAng - targAng
        elif diffPos < -0.75*PI:
            targAng = targAng - 2*PI
            diffPos = curAng - targAng

        Torque = Kp*diffPos + Kd*curVel
        return -Torque
        

    def ControlLoop(self):
        #radian = val/180.0*3.141592
        b = pendulum.hHinge
        #h = pendulum.hHinge
        torque = self.PD_controller(TARGET_ANGLE)
        #print torque
        b.addTorque(torque)
        

    def rotate(self, val):
        global TARGET_ANGLE
        TARGET_ANGLE = TARGET_ANGLE + val









