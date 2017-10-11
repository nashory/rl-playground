from mcode import *
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
import ode


# parameter setting.
BASE_H = 1.2
BASE_RADIUS = 0.35
MOTOR_H = 0.2
MOTOR_RADIUS = 0.25
H_BAR_RADIUS = MOTOR_RADIUS * 0.75
H_BAR_LEN = 1.3
H_BAR_THICKNESS = 0.10
V_BAR_RADIUS = MOTOR_RADIUS * 0.75
V_BAR_LEN = 1.5
V_BAR_THICKNESS = 0.10

GROUND_POS = (0.0, 0.0, 0.0)
MOTOR_POS_B = (0.0, BASE_H, 0.0)
MOTOR_POS_T = (0.0, BASE_H + H_BAR_THICKNESS, 0.0)
HBAR_POS_B = (0, BASE_H + H_BAR_THICKNESS/2, 0)
HBAR_POS_T = (-H_BAR_LEN, BASE_H + H_BAR_THICKNESS/2, 0)
VBAR_POS_B = add3(HBAR_POS_T, (-V_BAR_THICKNESS/2, 0.0, 0.0))
VBAR_POS_T = add3(VBAR_POS_B, (0.0, V_BAR_LEN, 0.0))

class Pendulum():
    def __init__(self, world, space, density, offset = (0.0, 0.0, 0.0)):
        """ Creates a ragdoll of standard size at the given offset. """
        self.world = world
        self.space = space
        self.density = density
        self.bodies = []
        self.geoms = []
        self.joints = []
        self.totalMass = 0.0

        self.offset = offset

        self.base = self.addBody(GROUND_POS, MOTOR_POS_B, BASE_RADIUS)
        self.dummy = self.addFixedJoint(self.base, ode.environment)
        self.motor = self.addBody(MOTOR_POS_B, MOTOR_POS_T, MOTOR_RADIUS, 'cylinder')
        self.hHinge = self.addHingeJoint(self.base, self.motor, MOTOR_POS_B, (0,1,0))
        #self.hMotor = self.addMotorJoint(self.base, self.motor, (0,1,0))
        self.hbar = self.addBody(HBAR_POS_B, HBAR_POS_T, H_BAR_RADIUS, 'hbar')
        self.hbar_mount = self.addFixedJoint(self.hbar, self.motor)
        self.vbar = self.addBody(VBAR_POS_B, VBAR_POS_T, V_BAR_RADIUS, 'vbar')
        self.vHinge = self.addHingeJoint(self.hbar, self.vbar, HBAR_POS_T, (1,0,0))



    def addBody(self, p1, p2, radius, type='cylinder'):
        p1 = add3(p1, self.offset)
        p2 = add3(p2, self.offset)
        length = dist3(p1, p2)

        body = ode.Body(self.world)
        m = ode.Mass()

        if type == 'cylinder':
            m.setCylinder(self.density, 3, radius, length)
            body.setMass(m)

            body.shape = "cylinder"
            body.length = length
            body.radius = radius

            geom = ode.GeomCylinder(self.space, radius, length)
            geom.setBody(body)
        if type == 'hbar':
            m.setBox(self.density, H_BAR_LEN, radius, H_BAR_THICKNESS)
        
            body.shape = "box"
            body.length = length
            body.width = radius
            body.height = H_BAR_THICKNESS
            body.len = H_BAR_LEN

            geom = ode.GeomBox(self.space, lengths=(H_BAR_LEN, radius, H_BAR_THICKNESS))
            geom.setBody(body)
        if type == 'vbar':
            m.setBox(self.density, V_BAR_LEN, radius, V_BAR_THICKNESS)
        
            body.shape = "box"
            body.length = length
            body.width = radius
            body.height = V_BAR_THICKNESS
            body.len = V_BAR_LEN

            geom = ode.GeomBox(self.space, lengths=(H_BAR_LEN, radius, H_BAR_THICKNESS))
            geom.setBody(body)


        # define body rotation automatically from body axis
        za = norm3(sub3(p2, p1))
        if (abs(dot3(za, (1.0, 0.0, 0.0))) < 0.7): xa = (1.0, 0.0, 0.0)
        else: xa = (0.0, 1.0, 0.0)
        ya = cross(za, xa)
        xa = norm3(cross(ya, za))
        ya = cross(za, xa)
        rot = (xa[0], ya[0], za[0], xa[1], ya[1], za[1], xa[2], ya[2], za[2])

        body.setPosition(mul3(add3(p1, p2), 0.5))
        body.setRotation(rot)

        self.bodies.append(body)
        self.geoms.append(geom)

        self.totalMass += body.getMass().mass

        return body


    def addFixedJoint(self, body1, body2):
        joint = ode.FixedJoint(self.world)
        joint.attach(body1, body2)
        joint.setFixed()

        joint.style = "fixed"
        self.joints.append(joint)

        return joint

    def addHingeJoint(self, body1, body2, anchor, axis):
        anchor = add3(anchor, self.offset)
        joint = ode.HingeJoint(self.world)
        joint.attach(body1, body2)
        joint.setAnchor(anchor)
        joint.setAxis(axis)

        joint.style = "hinge"
        self.joints.append(joint)

        return joint

    def addMotorJoint(self, body1, body2, axis):
        joint = ode.AMotor(self.world)
        joint.attach(body1, body2)
        joint.setNumAxes(1)
        #joint.setAnchor(anchor)
        joint.setAxis(0,1,axis)

        joint.style = "amotor"
        self.joints.append(joint)

        return joint

    
    def update(self):
        pass
   





