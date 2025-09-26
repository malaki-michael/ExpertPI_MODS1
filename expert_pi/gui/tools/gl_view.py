import numpy as np
import pyqtgraph.opengl as gl
from PySide6 import QtGui

shader = gl.shaders.ShaderProgram('edgeHilight', [
    gl.shaders.VertexShader("""
                varying vec3 normal;
                void main() {
                    // compute here for use in fragment shader
                    normal = normalize(gl_NormalMatrix * gl_Normal);
                    gl_FrontColor = gl_Color;
                    gl_BackColor = gl_Color;
                    gl_Position = ftransform();
                }
            """),
    gl.shaders.FragmentShader("""
                varying vec3 normal;
                void main() {
                    vec4 color = gl_Color;
                    float s = pow(normal.x*normal.x + normal.y*normal.y, 2.0);
                    if(s > 0.9)
                        s = 1;
                    else
                        s=0;
                    color.x = color.x + s * (1.0-color.x);
                    color.y = color.y + s * (1.0-color.y);
                    color.z = color.z + s * (1.0-color.z);
                    gl_FragColor = color;
                }
            """)
]),


class GlViewWidgetOrbit(gl.GLViewWidget):
    def __init__(self, orbit_callback):
        self.orbit_callback = orbit_callback
        super().__init__(rotationMethod="quaternion")

        self.linked_orbit = None
        self.slow_factor = 0.2

    def set_quaterion(self, q, ignore_link=False):
        self.opts['rotation'] = q
        self.update()
        if self.linked_orbit is not None and not ignore_link:
            self.linked_orbit.set_quaterion(q, ignore_link=True)
        mat = self.opts['rotation'].toRotationMatrix()
        self.orbit_callback(np.array(mat.data()).reshape(3, 3))

    def rotate(self, angle):
        q = QtGui.QQuaternion.fromEulerAngles(0, 0, angle)  # angle in deg
        q *= self.opts['rotation']
        self.set_quaterion(q)

    def orbit(self, azim, elev, ignore_link=False):
        super().orbit(self.slow_factor*azim, self.slow_factor*elev)

        mat = self.opts['rotation'].toRotationMatrix()

        if self.linked_orbit is not None and not ignore_link:
            self.linked_orbit.set_quaterion(self.opts['rotation'], ignore_link=True)
        self.orbit_callback(np.array(mat.data()).reshape(3, 3))
