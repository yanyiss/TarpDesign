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

import algorithm.newton_raphson as newton_raphson
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
        self.cam_target = np.array([0.0, 0.0, 3.45])
        self.cam_up_vector = np.array([0.0, 0.0, 1.0])
        
        self.rotate_speed = 0.01
        self.move_speed = 0.05

        self.color=np.array([1,0,0])
        self.point=np.array([0.0,0.0,0.0])
        #self.sign_one.connect(self.changecolor)

        """ self.df=deform.deform()
        self.faces=self.df.model.template_mesh.faces[0].clone().detach().cpu().numpy()
        self.df.set_init_force() """

        self.newton_raphson=newton_raphson.newton_raphson()
        self.gui_info=self.newton_raphson.gui_info

    
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
        vertices=self.gui_info.vertices
        faces=self.gui_info.faces
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

        #draw vertices grad
        if newton_raphson.params.use_vertgrad:
            #vertices_end=self.opt.simu_vertices_grad*1000+vertices
            vertices_end=self.gui_info.vertices_grad*0.01+vertices
            glColor3f(1.0,0.0,1.0)
            glLineWidth(2)
            glBegin(GL_LINES)
            for i in range(vertices_end.shape[0]):
                glVertex3f(vertices[i][0],vertices[i][1],vertices[i][2])
                glVertex3f(vertices_end[i][0],vertices_end[i][1],vertices_end[i][2])
            glEnd()
        
        boundary_index=self.gui_info.boundary_index
        rate=0.8
        forces=self.gui_info.forces*rate
        
        #draw current force
        id=0
        glLineWidth(2)
        glBegin(GL_LINES)
        glColor3f(0.7,0.0,0.0)
        for i in boundary_index:
            glVertex3f(vertices[i][0],vertices[i][1],vertices[i][2])
            glVertex3f(vertices[i][0]+forces[id][0],vertices[i][1]+forces[id][1],vertices[i][2]+forces[id][2])
            id=id+1
        glEnd()
        
        """ if newton_raphson.params.use_forcegrad:
            id=0
            forces_grad=self.gui_info.forces_grad*rate*50
            glLineWidth(2)
            glBegin(GL_LINES)
            glColor3f(0.0,0.0,0.8)
            for i in boundary_index:
                glVertex3f(vertices[i][0],vertices[i][1],vertices[i][2])
                glVertex3f(vertices[i][0]+forces_grad[id][0],vertices[i][1]+forces_grad[id][1],vertices[i][2]+forces_grad[id][2])
                id=id+1
            glEnd() """

        """ if newton_raphson.params.use_adamgrad:
            id=0
            equa_forces_grad=self.opt.simu_equa_force_grad*rate*1
            glLineWidth(2)
            glBegin(GL_LINES)
            glColor3f(0.0,0.8,0.0)
            for i in boundary_index:
                glVertex3f(vertices[i][0],vertices[i][1],vertices[i][2])
                glVertex3f(vertices[i][0]+equa_forces_grad[id][0],vertices[i][1]+equa_forces_grad[id][1],vertices[i][2]+equa_forces_grad[id][2])
                id=id+1
            glEnd() """

        #draw max force
        """ forcemax=(self.opt.tarp.tarp_info.Fmax*rate).clone().detach().cpu().numpy()
        id=0
        glLineWidth(2)
        glBegin(GL_LINES)
        glColor3f(1.0,0.0,0.0)
        for i in index:
            maxpos=forces[id]*forcemax/np.sqrt((forces[id,:]*forces[id,:]).sum())
            glVertex3f(vertices[i][0],vertices[i][1],vertices[i][2])
            glVertex3f(vertices[i][0]+maxpos[0],vertices[i][1]+maxpos[1],vertices[i][2]+maxpos[2])
            id=id+1
        glEnd() """

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
        delta_time = 0.001  # Time step
        """ if self.opt.itertimes>opt.params.max_iter-opt.params.updategl_hz:
            return
        if self.opt.small_gradient:
            return """
        """ for i in range(0,opt.params.updategl_hz):
            if self.opt.stop:
                return
            self.opt.one_iterate() """
        #self.opt.one_iterate()
        self.newton_raphson.one_iterate()
        self.update()
        

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
    
    """ def set_default(self):

        self.num_particles_x.setValue(10)
        self.num_particles_y.setValue(10)
        self.cloth_width.setValue(1.0)
        self.cloth_height.setValue(1.0)

        self.rotate_cloth_angle.setValue(90)
        self.rotate_cloth_axis.setCurrentText("X")

        self.translate_x.setValue(-0.5)
        self.translate_y.setValue(0.8)
        self.translate_z.setValue(-0.5)

        self.add_solid_check_box.setChecked(True)
        self.add_floor_check_box.setChecked(True)
        self.collision_object.setCurrentText("Cube")

        self.cube_size.setValue(0.5)
        self.cube_rotate_angle.setValue(0)
        self.cube_axis.setCurrentText("X")

        self.cube_translate_x.setValue(0.0)
        self.cube_translate_y.setValue(0.25)
        self.cube_translate_z.setValue(0.0)
        self.cube_scale_factor.setValue(1.0)

        self.sphere_radius.setValue(0.5)
        self.sphere_angle.setValue(0)
        self.sphere_axis.setCurrentText("X")

        self.sphere_translate_x.setValue(0.0)
        self.sphere_translate_y.setValue(0.25)
        self.sphere_translate_z.setValue(0.0)
        self.sphere_scale_factor.setValue(1.0)


        self.pyramid_height.setValue(0.5)
        self.pyramid_base.setValue(8)
        self.pyramid_radius.setValue(0.5)
        self.pyramid_angle.setValue(-90)
        self.pyramid_axis.setCurrentText("X")
        self.pyramid_translate_x.setValue(0.0)
        self.pyramid_translate_y.setValue(0.0)
        self.pyramid_translate_z.setValue(0.0)
        self.pyramid_scale.setValue(1.0)
    
        self.update_collision_object_fields() """

    """ def set_values_simulation(self):
        num_particles_x = self.num_particles_x.value()
        num_particles_y = self.num_particles_y.value()
        cloth_width = self.cloth_width.value()
        cloth_height = self.cloth_height.value()
        rotate_cloth_angle = self.rotate_cloth_angle.value()
        rotate_cloth_axis = AXIS_DICT[self.rotate_cloth_axis.currentText()]
        translate_x = self.translate_x.value()
        translate_y = self.translate_y.value()
        translate_z = self.translate_z.value()

        fix_first_line = self.fix_first_line.isChecked()

        cloth = Cloth(num_particles_x, num_particles_y, cloth_width, cloth_height, fix_first_line)
        cloth.rotate(rotate_cloth_axis, rotate_cloth_angle)
        cloth.translate([translate_x,translate_y,translate_z])

        current_item = self.collision_object.currentText()
        self.update_collision_object_fields()
        if current_item == "Cube":
            cube_size = self.cube_size.value()
            cube_rotate_angle = self.cube_rotate_angle.value()
            cube_axis = AXIS_DICT[self.cube_axis.currentText()]

            cube_translate_x = self.cube_translate_x.value()
            cube_translate_y = self.cube_translate_y.value()
            cube_translate_z = self.cube_translate_z.value()
            cube_scale_factor = self.cube_scale_factor.value()

            cube = Cube([0,0,0], cube_size)
            cube.rotate(cube_axis, cube_rotate_angle)
            cube.translate([cube_translate_x,cube_translate_y,cube_translate_z])
            cube.scale([cube_scale_factor,cube_scale_factor,cube_scale_factor])

            self.simulation_widget.solids = [cube]

        elif current_item == "Sphere":
            sphere_raidus = self.sphere_radius.value()
            sphere_rotate_angle = self.sphere_angle.value()
            sphere_axis = AXIS_DICT[self.sphere_axis.currentText()]

            sphere_translate_x = self.sphere_translate_x.value()
            sphere_translate_y = self.sphere_translate_y.value()
            sphere_translate_z = self.sphere_translate_z.value()
            sphere_scale_factor = self.sphere_scale_factor.value()

            sphere = Sphere([0,0,0], sphere_raidus)
            sphere.rotate(sphere_axis, sphere_rotate_angle)
            sphere.translate([sphere_translate_x,sphere_translate_y,sphere_translate_z])
            sphere.scale(sphere_scale_factor)
            self.simulation_widget.solids = [sphere]
        
        elif current_item == "Pyramid":
            pyramid_height = self.pyramid_height.value()
            pyramid_base = self.pyramid_base.value()
            pyramid_radius = self.pyramid_radius.value()
            pyramid_angle = self.pyramid_angle.value()
            pyramid_axis = AXIS_DICT[self.pyramid_axis.currentText()]
            pyramid_translate_x = self.pyramid_translate_x.value()
            pyramid_translate_y = self.pyramid_translate_y.value()
            pyramid_translate_z = self.pyramid_translate_z.value()
            pyramid_scale = self.pyramid_scale.value()

            pyramid = Pyramid([0.0, 0.0, 0.0], pyramid_base, pyramid_height, pyramid_radius)
            pyramid.rotate(pyramid_axis, pyramid_angle)
            pyramid.translate([pyramid_translate_x,pyramid_translate_y,pyramid_translate_z])
            pyramid.scale([pyramid_scale, pyramid_scale, pyramid_scale])

            self.simulation_widget.solids = [pyramid]

        self.simulation_widget.cloth = cloth """

    """ def toggle_combobox(self, checked):
        self.collision_object.setEnabled(checked)
        if checked:
            self.update_collision_object_fields()
        else :
            self.cube_group_box.setVisible(False)
            self.sphere_group_box.setVisible(False)
            self.pyramid_group_box.setVisible(False) """
    
    """ def toggle_floor(self, checked):
        if checked and self.plane not in self.simulation_widget.solids:
            self.simulation_widget.solids.append(self.plane)
        elif not checked and self.plane in self.simulation_widget.solids:
            self.simulation_widget.solids.remove(self.plane) """


    """ def update_collision_object_fields(self):
        current_item = self.collision_object.currentText()
        if current_item == "Cube":
            self.cube_group_box.setVisible(True)
            self.sphere_group_box.setVisible(False)
            self.pyramid_group_box.setVisible(False)
        elif current_item == "Sphere":
            self.cube_group_box.setVisible(False)
            self.sphere_group_box.setVisible(True)
            self.pyramid_group_box.setVisible(False)
        elif current_item == "Pyramid":
            self.cube_group_box.setVisible(False)
            self.sphere_group_box.setVisible(False)
            self.pyramid_group_box.setVisible(True) """

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

if __name__ == "__main__":
    app = QApplication(sys.argv)
    widget = MainWindow()
    widget.show()
    sys.exit(app.exec_())
