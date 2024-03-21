#!/usr/bin/env python

import sys
import time
import numpy as np
from PyQt5.QtCore import *
from PyQt5.QtWidgets import QApplication, QMainWindow, QOpenGLWidget, QVBoxLayout#, QWidget
from PyQt5 import uic

""" from Simulation.Cloth import Cloth
from Solids.Sphere import Sphere
from Solids.Cube import Cube
from Solids.Pyramid import Pyramid
from Solids.Plane import Plane """

from OpenGL.GL import *
from OpenGL.GLU import *

import algorithm.opt as opt
""" AXIS_DICT = {
    "X" : [1,0,0],
    "Y" : [0,1,0],
    "Z" : [0,0,1]
} """

class SimulationWidget(QOpenGLWidget):
    #sign_one = pyqtSignal()
    def initializeGL(self):
        glClearColor(0.5, 1.0, 1.0, 1.0)  # Set clear color to white
        glEnable(GL_DEPTH_TEST)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45.0, self.width() / self.height(), 0.1, 100.0)
        
        # Camera properties
        self.cam_position = np.array([0.0, -5.0, 5.0])
        self.cam_target = np.array([0.0, 0.0, 2.45])
        self.cam_up_vector = np.array([0.0, 0.0, 1.0])
        
        self.rotate_speed = 0.01
        self.move_speed = 0.05

        self.color=np.array([1,0,0])
        self.point=np.array([0.0,0.0,0.0])
        #self.sign_one.connect(self.changecolor)

        """ self.df=deform.deform()
        self.faces=self.df.model.template_mesh.faces[0].clone().detach().cpu().numpy()
        self.df.set_init_force() """

        self.opt=opt.deform()
        self.force_index=self.opt.simu_index

    
    def mousePressEvent(self, event):
        self.mouse_last_position = event.pos()
    
    def wheelEvent(self, event):
        delta = event.angleDelta().y() / 120  # The vertical scroll amount
        self.cam_position += (self.cam_target - self.cam_position) * self.move_speed * delta
        self.update()
        #self.sign_one.emit()

    def mouseMoveEvent(self, event):
        return
        dx = event.x() - self.mouse_last_position.x()
        dy = event.y() - self.mouse_last_position.y()

        #self.simulation_widget.cam_position += self.simulation_widget.move_speed * self.simulation_widget.cam_up_vector
        self.cam_target += self.rotate_speed * np.array([-dx, dy, 0])
        self.mouse_last_position = event.pos()

        self.update()
    
    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        gluLookAt(*(list(self.cam_position) + list(self.cam_target) + list(self.cam_up_vector)))


        #draw triangle mesh
        print('draw vertices')
        vertices=self.opt.tarp.vertices[0].clone().detach().cpu().numpy()
        faces=self.opt.tarp.faces[0].clone().detach().cpu().numpy()
        print('draw vertices done')
        glColor3f(1.0,1.0,0.0)
        glBegin(GL_TRIANGLES)
        for face in faces:
            glVertex3d(vertices[face[0]][0],vertices[face[0]][1],vertices[face[0]][2])
            glVertex3d(vertices[face[1]][0],vertices[face[1]][1],vertices[face[1]][2])
            glVertex3d(vertices[face[2]][0],vertices[face[2]][1],vertices[face[2]][2])
        glEnd() 

        glColor3f(0.0,0.0,0.0)
        glLineWidth(2)
        glBegin(GL_LINES)
        for face in faces:
            glVertex3f(vertices[face[0]][0],vertices[face[0]][1],vertices[face[0]][2])
            glVertex3f(vertices[face[1]][0],vertices[face[1]][1],vertices[face[1]][2])
            glVertex3f(vertices[face[2]][0],vertices[face[2]][1],vertices[face[2]][2])
            glVertex3f(vertices[face[0]][0],vertices[face[0]][1],vertices[face[0]][2])
            glVertex3f(vertices[face[1]][0],vertices[face[1]][1],vertices[face[1]][2])
            glVertex3f(vertices[face[2]][0],vertices[face[2]][1],vertices[face[2]][2])
        glEnd()
        
        # boundary_index=self.gui_info.boundary_index
        # rate=0.8
        # forces=self.gui_info.forces*rate
        

        #draw force 
        force=self.opt.force.now_force[0].clone().detach().cpu().numpy()
        id=0
        glLineWidth(2)
        glColor3f(0.7,0.0,0.0)
        glBegin(GL_LINES)
        for i in self.force_index:
            glVertex3f(vertices[i][0],vertices[i][1],vertices[i][2])
            glVertex3f(vertices[i][0]+force[id][0],vertices[i][1]+force[id][1],vertices[i][2]+force[id][2])
            id=id+1
        glEnd()


        #draw force grad
        force_grad=self.opt.force_grad[0].clone().detach().cpu().numpy()*50
        id=0
        glLineWidth(2)
        glColor3f(0.0,0.5,0.0)
        glBegin(GL_LINES)
        for i in self.force_index:
            glVertex3f(vertices[i][0],vertices[i][1],vertices[i][2])
            glVertex3f(vertices[i][0]+force_grad[id][0],vertices[i][1]+force_grad[id][1],vertices[i][2]+force_grad[id][2])
            id=id+1
        glEnd()

        #draw fff grad
        fff=self.opt.fff[0].clone().detach().cpu().numpy()*50
        id=0
        glLineWidth(2)
        glColor3f(0.5,0.5,0.0)
        glBegin(GL_LINES)
        for i in self.force_index:
            glVertex3f(vertices[i][0],vertices[i][1],vertices[i][2])
            glVertex3f(vertices[i][0]+fff[id][0],vertices[i][1]+fff[id][1],vertices[i][2]+fff[id][2])
            id=id+1
        glEnd()

        #draw vertices grad
        vertices_grad=self.opt.vertices_grad[0].clone().detach().cpu().numpy()*1000
        id=0
        glLineWidth(2)
        glColor3f(0.0,0.0,0.7)
        glBegin(GL_LINES)
        for i in range(vertices.shape[0]):
            glVertex3f(vertices[i][0],vertices[i][1],vertices[i][2])
            glVertex3f(vertices[i][0]+vertices_grad[i][0],vertices[i][1]+vertices_grad[i][1],vertices[i][2]+vertices_grad[i][2])
            id=id+1
        glEnd()

        

        #draw axis
        glLineWidth(2)
        glBegin(GL_LINES)
        glColor3f(1.0,0.0,0.0)
        glVertex3f(0.0,0.0,0.0)
        glVertex3f(10.0,0.0,0.0)
        glColor3f(0.0,1.0,0.0)
        glVertex3f(0.0,0.0,0.0)
        glVertex3f(0.0,10.0,0.0)
        glColor3f(0.0,0.0,1.0)
        glVertex3f(0.0,0.0,0.0)
        glVertex3f(0.0,0.0,10.0)
        glEnd()

        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        glFlush()

    def update_simulation(self):
        self.opt.one_iterate()
        

    def timerEvent(self, event):
        self.update_simulation()

class MainWindow(QMainWindow):
    def __init__(self, *args, **kwargs):
        super(QMainWindow, self).__init__(*args, **kwargs)

        uic.loadUi('ClothSimulationGUI/mainwindow.ui', self)

        self.simulation_widget = SimulationWidget(self.simulationWidget)
        #self.simulation_widget.solids = []
        self.timer = QTimer()

        """ self.plane = Plane()
        self.plane.rotate([1,0,0], -90)
        self.plane.scale(8.0)
        self.plane.translate([0.0, -0.01,0.0]) """

        self.timer.timeout.connect(self.simulation_widget.update_simulation)
        
        #self.collision_object.currentTextChanged.connect(self.update_collision_object_fields)

        self.simulation_widget.setFocusPolicy(Qt.StrongFocus)
        self.simulation_widget.setFocus()

        """ self.set_values_simulation()
        self.simulation_widget.solids.append(self.plane)
        self.add_solid_check_box.toggled.connect(self.toggle_combobox)
        self.add_floor_check_box.toggled.connect(self.toggle_floor) """

        layout = QVBoxLayout(self.simulationWidget)
        layout.addWidget(self.simulation_widget)

        #self.set_cloth_default_button.clicked.connect(self.set_default)
        self.simulate_button.clicked.connect(self.start_simulation)
        self.stop_simulation_button.clicked.connect(self.stop_simulation)
        self.restart_simulation_button.clicked.connect(self.restart_simulation) 

    def mousePressEvent(self, event):
        self.simulation_widget.mousePressEvent(event)

    def mouseMoveEvent(self, event):
        return
        self.simulation_widget.mouseMoveEvent(event)
    
    def timerEvent(self, event):
        self.simulation_widget.update_simulation()

    def keyPressEvent(self, event):
        # Move the camera depending on which key is pressed
        if event.key() == Qt.Key_Escape:
            self.close()
        if event.key() == Qt.Key_W:
            self.simulation_widget.cam_position += self.simulation_widget.move_speed * self.simulation_widget.cam_up_vector
        elif event.key() == Qt.Key_S:
            self.simulation_widget.cam_position -= self.simulation_widget.move_speed * self.simulation_widget.cam_up_vector
        elif event.key() == Qt.Key_A:
            self.simulation_widget.cam_position -= np.cross(self.simulation_widget.cam_target - self.simulation_widget.cam_position, self.simulation_widget.cam_up_vector) * self.simulation_widget.move_speed
        elif event.key() == Qt.Key_D:
            self.simulation_widget.cam_position += np.cross(self.simulation_widget.cam_target - self.simulation_widget.cam_position, self.simulation_widget.cam_up_vector) * self.simulation_widget.move_speed

        self.simulation_widget.update()

    def start_simulation(self):
        timer_interval = 1
        self.timer.start(timer_interval)

    def stop_simulation(self):
        if self.timer.isActive():
            self.timer.stop()
    
    def restart_simulation(self):
        self.stop_simulation()
        """ self.set_values_simulation()
        
        if not self.add_solid_check_box.isChecked() :
            self.simulation_widget.solids = []
            self.cube_group_box.setVisible(False)
            self.sphere_group_box.setVisible(False)
            self.pyramid_group_box.setVisible(False)
        if self.add_floor_check_box.isChecked() and self.plane not in self.simulation_widget.solids:
            self.simulation_widget.solids.append(self.plane) """
        self.start_simulation()

import torch
if __name__ == "__main__":
    app = QApplication(sys.argv)
    widget = MainWindow()
    widget.show()
    sys.exit(app.exec_())