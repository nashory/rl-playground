from mcode import *
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
import ode


# parameter setting.
UPPER_ARM_LEN = 0.30
FORE_ARM_LEN = 0.25
HAND_LEN = 0.13 # wrist to mid-fingers only
FOOT_LEN = 0.18 # ankles to base of ball of foot only
HEEL_LEN = 0.05

BROW_H = 1.68
MOUTH_H = 1.53
NECK_H = 1.50
SHOULDER_H = 1.37
CHEST_H = 1.35
HIP_H = 0.86
KNEE_H = 0.48
ANKLE_H = 0.08

SHOULDER_W = 0.41
CHEST_W = 0.36 # actually wider, but we want narrower than shoulders (esp. with large radius)
LEG_W = 0.28 # between middles of upper legs
PELVIS_W = 0.25 # actually wider, but we want smaller than hip width

R_SHOULDER_POS = (-SHOULDER_W * 0.5, SHOULDER_H, 0.0)
L_SHOULDER_POS = (SHOULDER_W * 0.5, SHOULDER_H, 0.0)
R_ELBOW_POS = sub3(R_SHOULDER_POS, (UPPER_ARM_LEN, 0.0, 0.0))
L_ELBOW_POS = add3(L_SHOULDER_POS, (UPPER_ARM_LEN, 0.0, 0.0))
R_WRIST_POS = sub3(R_ELBOW_POS, (FORE_ARM_LEN, 0.0, 0.0))
L_WRIST_POS = add3(L_ELBOW_POS, (FORE_ARM_LEN, 0.0, 0.0))
R_FINGERS_POS = sub3(R_WRIST_POS, (HAND_LEN, 0.0, 0.0))
L_FINGERS_POS = add3(L_WRIST_POS, (HAND_LEN, 0.0, 0.0))


R_HIP_POS = (-LEG_W * 0.5, HIP_H, 0.0)
L_HIP_POS = (LEG_W * 0.5, HIP_H, 0.0)
R_KNEE_POS = (-LEG_W * 0.5, KNEE_H, 0.0)
L_KNEE_POS = (LEG_W * 0.5, KNEE_H, 0.0)
R_ANKLE_POS = (-LEG_W * 0.5, ANKLE_H, 0.0)
L_ANKLE_POS = (LEG_W * 0.5, ANKLE_H, 0.0)
R_HEEL_POS = sub3(R_ANKLE_POS, (0.0, 0.0, HEEL_LEN))
L_HEEL_POS = sub3(L_ANKLE_POS, (0.0, 0.0, HEEL_LEN))
R_TOES_POS = add3(R_ANKLE_POS, (0.0, 0.0, FOOT_LEN))
L_TOES_POS = add3(L_ANKLE_POS, (0.0, 0.0, FOOT_LEN))


class Ragdoll():
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
        self.chest = self.addBody(  (-CHEST_W * 0.5, CHEST_H, 0.0),
                                    (CHEST_W * 0.5, CHEST_H, 0.0), 0.13)
        self.belly = self.addBody(  (0.0, CHEST_H - 0.1, 0.0),
                                    (0.0, HIP_H + 0.1, 0.0), 0.125)
        self.midSpine = self.addFixedJoint(self.chest, self.belly)
        self.pelvis = self.addBody( (-PELVIS_W * 0.5, HIP_H, 0.0),
                                    (PELVIS_W * 0.5, HIP_H, 0.0), 0.125)
        self.lowSpine = self.addFixedJoint(self.belly, self.pelvis)

        self.head = self.addBody(       (0.0, BROW_H, 0.0), (0.0, MOUTH_H, 0.0), 0.11)
        self.neck = self.addBallJoint(self.chest, self.head,
            (0.0, NECK_H, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, 1.0), pi * 0.1,
            pi * 0.2, 50.0, 35.0)

        self.rightUpperLeg = self.addBody(R_HIP_POS, R_KNEE_POS, 0.11)
        self.rightHip = self.addBallJoint(  self.pelvis, self.rightUpperLeg, R_HIP_POS,
                                            norm3((0.0, 0.0,-1.0)), (0.0, 0.0, 1.0),
                                            pi*0.05, pi*0.05, 100, 80)

        self.leftUpperLeg = self.addBody(L_HIP_POS, L_KNEE_POS, 0.11)
        self.leftHip = self.addBallJoint(   self.pelvis, self.leftUpperLeg, L_HIP_POS,
                                            norm3((0.0, 0.0,-1.0)), (0.0, 0.0, 1.0),
                                            pi*0.05, pi*0.05, 100, 80)

        
        self.rightLowerLeg = self.addBody(  R_KNEE_POS, R_ANKLE_POS, 0.09)

        self.rightKnee = self.addBallJoint( self.rightUpperLeg, self.rightLowerLeg, R_KNEE_POS,
                                            norm3((0.0, 0.0, 0.0)), (0.0, 1.0, 0.0),
                                            pi*0.05, pi*0.05, 80, 60) 

        self.leftLowerLeg = self.addBody(   L_KNEE_POS, L_ANKLE_POS, 0.09)
        self.leftKnee = self.addBallJoint(  self.leftUpperLeg, self.leftLowerLeg, L_KNEE_POS,
                                            norm3((0.0, 0.0, 0.0)), (0.0, 1.0, 0.0),
                                            pi*0.05, pi*0.05, 80, 60) 

        self.rightFoot = self.addBody(      R_HEEL_POS, R_TOES_POS, 0.09)
        self.rightAnkle = self.addBallJoint(self.rightLowerLeg, self.rightFoot, R_ANKLE_POS,
                                            norm3((0.0, 0.0, -1.0)), (0.0, 1.0, 0.0),
                                            pi*0.05, pi*0.05, 20, 10)
        self.leftFoot = self.addBody(       L_HEEL_POS, L_TOES_POS, 0.09)
        self.leftAnkle = self.addBallJoint( self.leftLowerLeg, self.leftFoot, L_ANKLE_POS,
                                            norm3((0.0, 0.0, -1.0)), (0.0, 1.0, 0.0),
                                            pi*0.05, pi*0.05, 20, 10) 
        

        self.rightUpperArm = self.addBody(R_SHOULDER_POS, R_ELBOW_POS, 0.08)
        self.rightShoulder = self.addBallJoint(self.chest, self.rightUpperArm, R_SHOULDER_POS, 
                                    norm3((-1.0, 2.0, 0.0)), (0.0, 0.0, 1.0), pi * 0.05,
                                    pi * 0.05, 150.0, 100.0)

        self.leftUpperArm = self.addBody(L_SHOULDER_POS, L_ELBOW_POS, 0.08)
        self.leftShoulder = self.addBallJoint(self.chest, self.leftUpperArm, L_SHOULDER_POS, 
                                    norm3((-1.0, 2.0, 0.0)), (0.0, 0.0, 1.0), pi * 0.05,
                                    pi * 0.05, 150.0, 100.0)

        self.rightForeArm = self.addBody(   R_ELBOW_POS, R_WRIST_POS, 0.075)
        self.rightElbow = self.addBallJoint(self.rightUpperArm, self.rightForeArm, R_ELBOW_POS,
                                            norm3((-1,0,-2)), (0.0,0.0,1.0),
                                            pi*0.05, pi*0.05, 50, 50)

        self.leftForeArm = self.addBody(    L_ELBOW_POS, L_WRIST_POS, 0.075)
        self.leftElbow = self.addBallJoint( self.leftUpperArm, self.leftForeArm, L_ELBOW_POS,
                                            norm3((1,0,-2)), (0.0,0.0,1.0),
                                            pi*0.05, pi*0.05, 50, 50)


        self.rightHand = self.addBody(      R_WRIST_POS, R_FINGERS_POS, 0.075)
        self.rightWrist = self.addBallJoint(self.rightForeArm,
                                            self.rightHand, R_WRIST_POS)
        self.leftHand = self.addBody(       L_WRIST_POS, L_FINGERS_POS, 0.075)
        self.leftWrist = self.addBallJoint( self.leftForeArm,
                                            self.leftHand, L_WRIST_POS)



    def addBody(self, p1, p2, radius):
        """
        adds a capsule body between joint posotions p1 and p2 with given
        radius to the ragdoll.
        """
        p1 = add3(p1, self.offset)
        p2 = add3(p2, self.offset)

        # cylinder length not including endcaps, make capsules overlap by half
        #   radius at joints
        cyllen = dist3(p1, p2) - radius

        body = ode.Body(self.world)
        m = ode.Mass()
        m.setCappedCylinder(self.density, 3, radius, cyllen)
        body.setMass(m)

        # set parameters for drawing the body
        body.shape = "capsule"
        body.length = cyllen
        body.radius = radius

        # create a capsule geom for collision detection
        geom = ode.GeomCCylinder(self.space, radius, cyllen)
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
        joint = ode.HingeJoint(self.world)
        joint.attach(body1, body2)
        joint.setAnchor(anchor)
        joint.setAxis(axis)

        joint.style = "hinge"
        self.joints.append(joint)

        return joint




    def addBallJoint(self, body1, body2, anchor, baseAxis=None, baseTwistUp=None,
        flexLimit = pi, twistLimit = pi, flexForce = 0.0, twistForce = 0.0):
        
        def getBodyRelVec(b, v):
            """
            Returns the 3-vector v transformed into the local coordinate system of ODE
            body b.
            """
            return rotate3(invert3x3(b.getRotation()), v)
        

        anchor = add3(anchor, self.offset)

        # create the joint
        joint = ode.BallJoint(self.world)
        joint.attach(body1, body2)
        joint.setAnchor(anchor)

        if baseAxis is not None:
            joint.haveAxis = True
            # store the base orientation of the joint in the local coordinate system
            #   of the primary body (because baseAxis and baseTwistUp may not be
            #   orthogonal, the nearest vector to baseTwistUp but orthogonal to
            #   baseAxis is calculated and stored with the joint)
            joint.baseAxis = getBodyRelVec(body1, baseAxis)
            tempTwistUp = getBodyRelVec(body1, baseTwistUp)
            baseSide = norm3(cross(tempTwistUp, joint.baseAxis))
            joint.baseTwistUp = norm3(cross(joint.baseAxis, baseSide))

            # store the base twist up vector (original version) in the local
            #   coordinate system of the secondary body
            joint.baseTwistUp2 = getBodyRelVec(body2, baseTwistUp)

            # store joint rotation limits and resistive force factors
            joint.flexLimit = flexLimit
            joint.twistLimit = twistLimit
            joint.flexForce = flexForce
            joint.twistForce = twistForce
        else:
            joint.haveAxis = False
       
        joint.style = "ball"
        self.joints.append(joint)

        return joint

    
    def update(self):
        for j in self.joints:
            if j.style == "ball" and j.haveAxis:
                # determine base and current attached body axes
                baseAxis = rotate3(j.getBody(0).getRotation(), j.baseAxis)
                currAxis = zaxis(j.getBody(1).getRotation())

                # get angular velocity of attached body relative to fixed body
                relAngVel = sub3(j.getBody(1).getAngularVel(),
                    j.getBody(0).getAngularVel())
                twistAngVel = project3(relAngVel, currAxis)
                flexAngVel = sub3(relAngVel, twistAngVel)

                # restrict limbs rotating too far from base axis
                angle = acosdot3(currAxis, baseAxis)
                if angle > j.flexLimit:
                    # add torque to push body back towards base axis
                    j.getBody(1).addTorque(mul3(
                        norm3(cross(currAxis, baseAxis)),
                        (angle - j.flexLimit) * j.flexForce))

                    # dampen flex to prevent bounceback
                    j.getBody(1).addTorque(mul3(flexAngVel,
                        -0.01 * j.flexForce))

                # determine the base twist up vector for the current attached
                #   body by applying the current joint flex to the fixed body's
                #   base twist up vector
                baseTwistUp = rotate3(j.getBody(0).getRotation(), j.baseTwistUp)
                base2current = calcRotMatrix(norm3(cross(baseAxis, currAxis)),
                    acosdot3(baseAxis, currAxis))
                projBaseTwistUp = rotate3(base2current, baseTwistUp)

                # determine the current twist up vector from the attached body
                actualTwistUp = rotate3(j.getBody(1).getRotation(),
                    j.baseTwistUp2)

                # restrict limbs twisting
                angle = acosdot3(actualTwistUp, projBaseTwistUp)
                if angle > j.twistLimit:
                    # add torque to rotate body back towards base angle
                    j.getBody(1).addTorque(mul3(
                        norm3(cross(actualTwistUp, projBaseTwistUp)),
                        (angle - j.twistLimit) * j.twistForce))

                    # dampen twisting
                    j.getBody(1).addTorque(mul3(twistAngVel,
                        -0.01 * j.twistForce))

   





