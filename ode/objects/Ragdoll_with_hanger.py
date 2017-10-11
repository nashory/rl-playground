from mcode import *
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
import ode
import Ragdoll_simple as doll


# parameter settings for hanger.
H_BAR_LEN = 2.4
V_BAR_LEN = 2.4
H_BAR_RADIUS = 0.08
V_BAR_RADIUS = 0.16
BASE_SIZE = 5
BASE_THICKNESS = 0.05
BASE_DENSITY = 100000
GRAB_OFFSET = 0.5
GRAB_SIZE = 0.1


class Ragdoll_with_hanger():
    def __init__(self, world, space, density, offset = (0.0, 0.0, 0.0)):
        """ Creates a ragdoll with hanger. """
        self.world = world
        self.space = space
        self.density = density
        self.bodies = []
        self.geoms = []
        self.joints = []
        self.totalMass = 0.0

        self.offset = offset
        self.base = self.addBody('vertical',    (0.0, 0.0, 0.0),
                                                (0.0, BASE_THICKNESS, 0.0),
                                                BASE_SIZE)
        self.left_v_bar = self.addBody('vertical',  (-H_BAR_LEN*0.5, 0.0, 0.0), 
                                                    (-H_BAR_LEN*0.5, V_BAR_LEN, 0.0), V_BAR_RADIUS)
        self.right_v_bar = self.addBody('vertical', (H_BAR_LEN*0.5, 0.0, 0.0), 
                                                    (H_BAR_LEN*0.5, V_BAR_LEN, 0.0), V_BAR_RADIUS)
        self.h_bar = self.addBody('horizontal', (-H_BAR_LEN*0.5, V_BAR_LEN, 0.0), 
                                                (H_BAR_LEN*0.5, V_BAR_LEN, 0.0), H_BAR_RADIUS)
        
        self.left_base = self.addFixedJoint(self.left_v_bar, self.base)
        self.right_base = self.addFixedJoint(self.right_v_bar, self.base)

        self.left_hinge = self.addFixedJoint(self.left_v_bar, self.h_bar)
        self.right_hinge = self.addFixedJoint(self.right_v_bar, self.h_bar)

        self.doll = doll.Ragdoll(world, space, 500, (0.0, V_BAR_LEN - doll.R_FINGERS_POS[1]-0.5, 0.0))
        print "total mass of ragdoll is %.1f kg (%.1f lbs)" % (self.doll.totalMass, self.doll.totalMass * 2.2)
       

        self.doll.leftHand.setPosition((-0.24,V_BAR_LEN,0)) 
        self.doll.rightHand.setPosition((0.24,V_BAR_LEN,0)) 
        self.left_grap = self.addHingeJoint(self.doll.leftHand, self.h_bar, (0, V_BAR_LEN, 0), (-1,0,0))
        self.right_grap = self.addHingeJoint(self.doll.rightHand, self.h_bar, (0, V_BAR_LEN, 0), (-1,0,0))
    

        # insert ragdoll parts into our bodies list.
        for b in self.doll.bodies:
            self.bodies.append(b)
       
 
    def addBody(self, type, p1, p2, radius):
        
        p1 = add3(p1, self.offset)
        p2 = add3(p2, self.offset)
        length = dist3(p1, p2)

        body = ode.Body(self.world)
        m = ode.Mass()
        if type == 'horizontal':
            m.setCylinder(self.density, 3, radius, length)
            body.setMass(m)

            body.shape = "capsule"
            body.length = length
            body.radius = radius

            geom = ode.GeomCylinder(self.space, radius, length)
            geom.setBody(body)


        elif type == 'vertical':
            m.setBox(self.density, radius, radius, length)
            body.setMass(m)
    
            body.shape = "box"
            body.length = length
            body.radius = radius
            
            geom = ode.GeomBox(self.space, lengths=(radius, radius, length))
            geom.setBody(body)
        
        elif type == 'base':
            m.setBox(BASE_DENSITY, radius, radius, length)
            body.setMass(m)
    
            body.shape = "box"
            body.length = length
            body.radius = radius
            
            geom = ode.GeomBox(self.space, lengths=(radius, radius, length))
            geom.setBody(body)


            
        # define body rotation automatically from body axis.
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
        joint = ode.HingeJoint(self.world)
        joint.attach(body1, body2)
        joint.setAnchor(anchor)
        joint.setAxis(axis)

        joint.style = "hinge"
        self.joints.append(joint)

        return joint


    def draw(self):
        CAPSULE_SLICES = 16
        CAPSULE_STACKS = 12
       
        print self.bodies 
        for body in self.bodies:
            rot = makeOpenGLMatrix(body.getRotation(), body.getPosition())
            glPushMatrix()
            glMultMatrixd(rot)
            if body.shape == 'capsule':
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
            elif body.shape == 'box':
                sx, sy, sz = body.radius, body.radius, body.length
                glscalef(sx, sy, sz)
                glutSolidCube(1)
            glPopMatrix()




    def update(self):
       self.doll.update() 
            





