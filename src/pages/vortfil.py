import dash
from dash import html, dcc
import dash_bootstrap_components as dbc
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

import time

import plotly.express as px

import scipy.spatial.transform as transform_sp

class VortexFilament:
    def __init__(self, strength, point, vector,start=None, end=None, prop = None):
        self.children=[]
        self.parent = None
        self.Strength = strength
        self.Point = np.array(point)
        self.vector = np.array(vector)
        self.vector = self.vector/np.linalg.norm(self.vector)
        if start is not None:
            self.start = np.array(start)
        else:
            self.start = start
        if end is not None:
            self.end = np.array(end)
        else:
            self.end = end
        self.prop = prop

        self.new_vector = None
        self.new_Point = None
        self.new_start = None
        self.new_end = None

    def calculate_velocity(self, coord):
        h, dir= self.calculate_h(coord)
        if self.start is not None:
            starting = np.dot(coord-self.start,self.vector)/(np.linalg.norm(coord-self.start)*np.linalg.norm(self.vector))
        if self.end is not None:
            ending = np.dot(coord-self.end,self.vector)/(np.linalg.norm(coord-self.end)*np.linalg.norm(self.vector))
        
        if self.start is None and self.end is None:
            return self.Strength/(2*np.pi*h), dir
        elif self.start is not None and self.end is None:
            return self.Strength/(4*np.pi*h)*(starting+1), dir
        elif self.end is not None and self.start is None:
            return self.Strength/(4*np.pi*h)*(1-ending), dir
        else:
            return self.Strength/(4*np.pi*h)*(starting-ending), dir
        
    def calculate_velocity_2(self, coord):
        coord = np.array(coord)
        if self.start is None and self.end is None:
            dir  = np.cross(self.vector,coord-self.Point)
            dir_norm = dir/np.linalg.norm(dir)
            h = np.linalg.norm(dir)/np.linalg.norm(self.vector)
            return self.Strength/(2*np.pi*h)* dir_norm
            
        elif self.start is not None and self.end is None:
            r1 = self.start-coord
            r1norm = np.linalg.norm(r1)
            r2 = self.vector
            r2norm = np.linalg.norm(r2)
            r1r2cross = np.cross(r1,r2)
            crossnorm = np.linalg.norm(r1r2cross)
            return self.Strength/(4*np.pi)*r1r2cross/(crossnorm*crossnorm)*(np.dot((-r2), r1)/r1norm+r2norm)
        elif self.start is None and self.end is not None:
            r1 = -self.vector
            r1norm = np.linalg.norm(r1)
            r2 = self.end - coord
            r2norm = np.linalg.norm(r2)
            r1r2cross = np.cross(r1,r2)
            crossnorm = np.linalg.norm(r1r2cross)
            return self.Strength/(4*np.pi)*r1r2cross/(crossnorm*crossnorm)*(r1norm-np.dot((r1), r2)/r2norm)
        else:
            r1 = self.start-coord
            r1norm = np.linalg.norm(r1)
            r2 = self.end - coord
            r2norm = np.linalg.norm(r2)
            r1r2cross = np.cross(r1,r2)
            crossnorm = np.linalg.norm(r1r2cross)
            return self.Strength/(4*np.pi)*r1r2cross/(crossnorm*crossnorm)*(np.dot((r1-r2), r1/r1norm-r2/r2norm))
        
    def calculate_h(self, coord):
        AP = np.array(coord)-self.Point
        cross_product = np.cross(self.vector,AP)
        cross_product_normed = cross_product/np.linalg.norm(cross_product)
        sign = np.sign(cross_product[2])
        return np.linalg.norm(cross_product)/(np.linalg.norm(self.vector)), cross_product_normed
    
    def calculate_velocity_family(self, coord):
        mag, dir = self.calculate_velocity(coord)
        vel = mag*dir
        for i in self.children:
            vel += i.calculate_velocity_family(coord)
        return vel
    def calculate_velocity_family_2(self, coord):
        check = self.check_point_on_line(coord)
        if check == True:
            print("Point on line")
            vel = np.array([0.,0.,0.])
        else:
            vel = self.calculate_velocity_2(coord)

        for i in self.children:
            vel += i.calculate_velocity_family_2(coord)
        return vel
    
    def eq_of_filament(self,mu):
        return self.point+mu*self.vector
    def check_point_on_line(self, coord):
        mu=np.zeros(3)
        Threshold = 0.1
        if 0 in self.vector:
            zero_element = np.where(np.round(self.vector,7) == 0)[0]
            non_zero_element = np.where(np.round(self.vector,7) != 0)[0]
            for i in non_zero_element:
                mu[i] = (coord[i]-self.Point[i])/self.vector[i]
            for i in zero_element:
                if abs(coord[i] -self.Point[i])>Threshold:
                    return False
                else:
                    mu[i]=mu[np.random.choice(non_zero_element)]
        else:
            mu = coord-self.Point/self.vector
        
        if abs(mu[0] - mu[1]) > Threshold or abs(mu[0] - mu[2]) > Threshold or abs(mu[1] - mu[2]) > Threshold:
            return False
        return True
    def split(self, Strengths, coord, angles):
        self.check_point_on_line(coord)
        coord = np.array(coord)
        self.end = coord

        Strengths = np.array(Strengths)
        for i in range(len(angles)):
            angles[i]=angles[i]*np.pi/180
            vec = np.array([self.vector[0]*np.cos(angles[i])-self.vector[1]*np.sin(angles[i]),self.vector[0]*np.sin(angles[i])+self.vector[1]*np.cos(angles[i]),0])*np.sign(Strengths[i])
            if Strengths[i] < 0: 
                self.children.append(VortexFilament(abs(Strengths[i])*self.Strength, coord, vec, end = coord,prop = Strengths[i]))
            else:
                self.children.append(VortexFilament(Strengths[i]*self.Strength, coord, vec, start = coord,prop = abs(Strengths[i])))

        print(self.Strength, self.Point, self.vector, self.start, self.end)
        print(self.children[0].Strength, self.children[0].Point, self.children[0].vector, self.children[0].start, self.children[0].end)
        print(self.children[1].Strength, self.children[1].Point, self.children[1].vector, self.children[1].start, self.children[1].end)
        for i in self.children:
            i.parent = self

    def bend(self, coord, angle):
        self.check_point_on_line(coord)
        coord = np.array(coord)
        self.end = coord
        angle=angle*np.pi/180
        vec = np.array([self.vector[0]*np.cos(angle)-self.vector[1]*np.sin(angle),self.vector[0]*np.sin(angle)+self.vector[1]*np.cos(angle),0])
        self.children.append(VortexFilament(self.Strength, coord, vec, start = coord))
        for i in self.children:
            i.parent = self

    def calc_vortex_family(self,parent_strength = None):
        if self.prop is not None and parent_strength is not None:
            self.Strength = self.prop*parent_strength
        for i in self.children:
            i.calc_vortex_family(self.Strength)

    def draw_all(self, limit_x, limit_y, limit_z, y_val=-1, reso = 52):
        fig = self.draw_vortex_family_3d(limit_x, limit_y, limit_z)
        # fig = self.draw_surface(limit_x, limit_y, limit_z, y_val, fig, reso)
        start = time.time()
        fig = self.draw_cones(limit_x, limit_y, limit_z, fig, 10)
        print("Time to draw cones: ", time.time()-start)
        return fig

    def draw_surface(self, limit_x, limit_y, limit_z, y_val=-1, fig = None, num = 52):
        start = time.time()
        self.X, self.Y, self.Z, self.vel = self.get_surface(limit_x, limit_z, y_val, num_points=num)
        print("Time to calculate surface: ", time.time()-start)

        if fig is None:
            fig = go.Figure()
            fig.update_layout(
                width=800,
                height=700,
                autosize=False,
                scene=dict(
                    xaxis=dict(range=[limit_x[0], limit_x[1]],
                               backgroundcolor="rgb(200, 230,200)",
                               gridcolor="white",
                               showbackground=True,
                               zerolinecolor="white"),
                    yaxis=dict(range=[limit_y[0], limit_y[1]],
                               backgroundcolor="rgb(230, 200,200)",
                               gridcolor="white",
                               showbackground=True,
                               zerolinecolor="white"),
                    zaxis=dict( range=[limit_z[0], limit_z[1]],
                                backgroundcolor="rgb(200, 200, 230)",
                                gridcolor="white",
                                showbackground=True,
                                zerolinecolor="white"),
                    camera=dict(
                        up=dict(
                            x=0,
                            y=0,
                            z=1
                        ),
                        eye=dict(
                            x=-1,
                            y=-0.8,
                            z=0.8,
                        )
                    ),
                    aspectratio = dict( x=1, y=1, z=0.7 ),
                    aspectmode = 'manual'
                ),
            )
        fig.add_trace(go.Surface(x=self.X, y=self.Y, z=self.Z, surfacecolor=self.vel, colorscale='Viridis'))

        return fig

    def gather_points(self, limit_x, limit_y, limit_z, dist = 0.1):
        hor = np.cross(self.vector, [0,0,1])
        hor = hor/np.linalg.norm(hor)

        first_point = np.array([self.limit[0],self.limit[2], 0])

        x_points = np.array([])
        y_points = np.array([])
        z_points = np.array([])

        num=3

        num_angles = 6
        
        r_list = np.zeros(num)

        for i in range(num):
            r_list[i] = dist*1.8**i
        sep = self.line_length/self.elem_size
        
        for i in range(num):
            for j in range(num_angles):
                new_point = first_point + r_list[i]*np.cos(j*2*np.pi/num_angles)*hor + r_list[i]*np.sin(j*2*np.pi/num_angles)*np.array([0,0,1])
                x_points = np.concatenate((x_points,[new_point[0]]))
                y_points = np.concatenate((y_points,[new_point[1]]))
                z_points = np.concatenate((z_points,[new_point[2]]))


            # up = first_point + r_list[i]*hor
            # down = first_point - r_list[i]*hor
            # left = first_point + r_list[i]*np.array([0,0,1])
            # right = first_point - r_list[i]*np.array([0,0,1])

            # x_points = np.concatenate((x_points,[up[0],down[0],left[0],right[0]]))
            # y_points = np.concatenate((y_points,[up[1],down[1],left[1],right[1]]))
            # z_points = np.concatenate((z_points,[up[2],down[2],left[2],right[2]]))

                for k in range(self.elem_size):
                    new_point += self.vector*sep
                    x_points = np.concatenate((x_points,[new_point[0]]))
                    y_points = np.concatenate((y_points,[new_point[1]]))
                    z_points = np.concatenate((z_points,[new_point[2]]))

        for i in self.children:
            child_x, child_y, child_z = i.gather_points(limit_x, limit_y, limit_z, dist)
            x_points = np.concatenate((x_points,child_x))
            y_points = np.concatenate((y_points,child_y))
            z_points = np.concatenate((z_points,child_z))

        
        return x_points, y_points, z_points
        





    def draw_cones(self, limit_x, limit_y, limit_z, fig=None, reso = 30):
        X,Y,Z = self.gather_points(limit_x, limit_y, limit_z)

        # fig_plt = plt.figure()
        # ax = fig_plt.add_subplot(111, projection='3d')
        # ax.scatter(X,Y,Z)
        # ax.set_xlim(limit_x[0], limit_x[1])
        # ax.set_ylim(limit_y[0], limit_y[1])
        # ax.set_zlim(limit_z[0], limit_z[1])
        # plt.show()

        x_ = np.linspace(limit_x[0], limit_x[1], reso)
        y_ = np.linspace(limit_y[0], limit_y[1], reso)
        z_ = np.linspace(limit_z[0], limit_z[1], reso)

        X_grid, Y_grid, Z_grid = np.meshgrid(x_, y_, z_)

        X = np.concatenate((X,X_grid.flatten()))
        Y = np.concatenate((Y,Y_grid.flatten()))
        Z = np.concatenate((Z,Z_grid.flatten()))

        u = np.zeros(X.shape)
        v = np.zeros(X.shape)
        w = np.zeros(X.shape)

        for i in range(X.shape[0]):
            coord = np.array([X[i],Y[i],Z[i]])
            vel = self.calculate_velocity_family_2(coord)
            u[i] = vel[0]
            v[i] = vel[1]
            w[i] = vel[2]
            
        fig.add_trace(go.Cone(x=X, y=Y, z=Z, u=u, v=v, w=w, showscale=True, colorscale='Viridis', sizemode='scaled', sizeref=2))

        return fig
        
        
        

    def draw_vortex_family_3d(self, limit_x, limit_y, limit_z, fig=None,res = 0.1):
        self.new_Point = self.Point
        self.new_vector = self.vector


        box = self.calculate_box_intersection(limit_x, limit_y)
        if self.start is not None and self.end is not None:
            limit = np.array([self.start[0],self.end[0],self.start[1],self.end[1]])
        elif self.start is not None and self.end is None:
            limit = np.array([self.start[0],box[1][0],self.start[1],box[1][1]])
        elif self.start is None and self.end is not None:
            limit = np.array([box[0][0],self.end[0],box[0][1],self.end[1]])
        else:
            limit = np.array([box[0][0],box[1][0],box[0][1],box[1][1]])
        
        self.line_length = np.sqrt((limit[1]-limit[0])**2+(limit[3]-limit[2])**2)
        self.elem_size = int(np.floor(self.line_length/res)+1)

        if self.elem_size < 2:
            self.elem_size = 2

        self.limit = limit

        x_arr = np.linspace(limit[0], limit[1], self.elem_size)
        y_arr = np.linspace(limit[2], limit[3], self.elem_size)
        z_arr = np.zeros(x_arr.shape)
        
        if fig is None:
            fig = go.Figure()
            fig.update_layout(
                width=800,
                height=700,
                autosize=False,
                scene=dict(
                    xaxis=dict(range=[limit_x[0], limit_x[1]],
                               backgroundcolor="rgb(200, 230,200)",
                               gridcolor="white",
                               showbackground=True,
                               zerolinecolor="white"),
                    yaxis=dict(range=[limit_y[0], limit_y[1]],
                               backgroundcolor="rgb(230, 200,200)",
                               gridcolor="white",
                               showbackground=True,
                               zerolinecolor="white"),
                    zaxis=dict( range=[limit_z[0], limit_z[1]],
                                backgroundcolor="rgb(200, 200, 230)",
                                gridcolor="white",
                                showbackground=True,
                                zerolinecolor="white"),
                    camera=dict(
                        up=dict(
                            x=0,
                            y=0,
                            z=1
                        ),
                        eye=dict(
                            x=-1,
                            y=-0.8,
                            z=0.8,
                        )
                    ),
                    aspectratio = dict( x=1, y=1, z=0.7 ),
                    aspectmode = 'manual'
                ),
            )
        fig.add_trace(go.Scatter3d(
                x=x_arr, y=y_arr, z=z_arr,
                marker=dict(
                    size=0
                ),
                line=dict(
                    color='darkblue',
                    width=2
                )
            ))
        
        
        for i in self.children:
            i.draw_vortex_family_3d(limit_x, limit_y, limit_z, fig)

        return fig

    def get_surface(self, limit_x, limit_z,y_val,  num_points = 100):
        self.calc_vortex_family()
        x_arr = np.linspace(limit_x[0], limit_x[1], num_points)
        z_arr = np.linspace(limit_z[0], limit_z[1], num_points)

        X, Z = np.meshgrid(x_arr, z_arr)
        print(X.shape)
        print(X)
        vel = np.zeros(X.shape)
        for i in range(num_points):
            for j in range(num_points):
                coord = np.array([X[i,j],y_val,Z[i,j]])
                vel[i,j] = np.linalg.norm(self.calculate_velocity_family_2(coord))

        Y = np.ones(X.shape)*y_val
        return X, Y, Z, vel

        
    def draw_vortex_family(self, limit_x, limit_y,fig=None, transform = None):
        if transform is not None:
            self.new_vector = np.dot(transform,self.vector)
            self.new_Point = np.dot(transform,self.Point)

            if self.start is not None:
                self.new_start = np.dot(transform,self.start)
            if self.end is not None:
                self.new_end = np.dot(transform,self.end)
        else:
            self.new_vector = self.vector
            self.new_Point = self.Point
            if self.start is not None:
                self.new_start = self.start
            if self.end is not None:
                self.new_end = self.end
        
        
        self.calc_vortex_family(self.Strength)
        box = self.calculate_box_intersection(limit_x, limit_y)
        if self.start is not None and self.end is not None:
            limit = np.array([self.new_start[0],self.new_end[0],self.new_start[1],self.new_end[1]])
        elif self.start is not None and self.end is None:
            limit = np.array([self.new_start[0],box[1][0],self.new_start[1],box[1][1]])
        elif self.start is None and self.end is not None:
            limit = np.array([box[0][0],self.new_end[0],box[0][1],self.new_end[1]])
        else:
            limit = np.array([box[0][0],box[1][0],box[0][1],box[1][1]])
            
        # Add a trace for the straight line
        x_arr = np.linspace(limit[0], limit[1], 4)
        y_arr = np.linspace(limit[2], limit[3], 4)

        plt.plot(x_arr,y_arr)
        
        if fig is None: 
            fig = go.Figure()        
            fig.update_layout(xaxis_title='X-axis',
                              yaxis_title='Y-axis',
                              title='Straight Line Plot'
                              )
            # fig.update_xaxes(range=[limit_x[0]*transform[0,0], limit_x[1]*transform[0,0]])  # Set x-axis limits
            # fig.update_yaxes(range=[limit_y[0]*transform[1,1], limit_y[1]*transform[1,1]])  # Set y-axis limits

            fig.update_xaxes(range=[limit_x[0], limit_x[1]])  # Set x-axis limits
            fig.update_yaxes(range=[limit_y[0], limit_y[1]])  # Set y-axis limits

            limit_interval_x = limit_x[1]-limit_x[0]
            limit_interval_y = limit_y[1]-limit_y[0]

            # grids_x = np.linspace(limit_x[0], limit_x[1], int(limit_interval_x))
            # grids_y = np.linspace(limit_y[0], limit_y[1], int(limit_interval_y))
            # for i in grids_x:
            #     x_arr_grid = ([i*transform[0,0]-10*transform[0,1], i*transform[0,0]+10*transform[0,1]])
            #     y_arr_grid = ([-10*transform[1,1]+i*transform[1,0], 10*transform[1,1]+i*transform[1,0]])
                
            #     fig.add_shape(type="line", x0=x_arr_grid[0], y0=y_arr_grid[0], x1=x_arr_grid[1], y1=y_arr_grid[1], line=dict(color="grey", width=0.5))
            # for i in grids_y:

            #     x_arr_grid = ([-10*transform[0,0]+i*transform[0,1], 10*transform[0,0]+i*transform[0,1]])
            #     y_arr_grid = ([i*transform[1,1]-10*transform[1,0], i*transform[1,1]+10*transform[1,0]])

            #     fig.add_shape(type="line", x0=x_arr_grid[0], y0=y_arr_grid[0], x1=x_arr_grid[1], y1=y_arr_grid[1], line=dict(color="grey", width=0.5))
        

        fig.add_trace(go.Scatter(x=x_arr, y=y_arr,
                                mode='lines', 
                                line=dict(color='blue', width=2),
                                name='Straight Line'))
        
        # Set axis labels

        for i in self.children:
            i.draw_vortex_family(limit_x, limit_y, fig, transform)
        return fig

    def calculate_box_intersection(self, limit_x, limit_y):
        box_intersection = []
        y_intersects = []
        x_intersects = []
        if self.new_vector is not None:
            for i in limit_x:
                if (np.round(self.new_vector[0],6) == 0):
                    y_intersects.append(i*100)
                else:
                    y_intersects.append((i-self.new_Point[0])/self.new_vector[0]*self.new_vector[1]+self.new_Point[1])
            for i in limit_y:
                if (np.round(self.new_vector[1],6) == 0):
                    x_intersects.append(i*100)
                else:
                    x_intersects.append((i-self.new_Point[1])/self.new_vector[1]*self.new_vector[0]+self.new_Point[0])
        else:
            for i in limit_x:
                if self.vector[0] == 0:
                    y_intersects.append(0)
                else:
                    y_intersects.append((i-self.Point[0])/self.vector[0]*self.vector[1]+self.Point[1])
            for i in limit_y:
                if self.vector[1] == 0:
                    x_intersects.append(0)
                else:
                    x_intersects.append((i-self.Point[1])/self.vector[1]*self.vector[0]+self.Point[0])

        y_intersects = np.round(y_intersects,6)
        x_intersects = np.round(x_intersects,6)

        if y_intersects[0]>=limit_y[0] and y_intersects[0]<=limit_y[1]:
            box_intersection.append([limit_x[0],y_intersects[0]])
        if y_intersects[1]>=limit_y[0] and y_intersects[1]<=limit_y[1]:
            box_intersection.append([limit_x[1],y_intersects[1]])
        if x_intersects[0]>=limit_x[0] and x_intersects[0]<=limit_x[1]:
            box_intersection.append([x_intersects[0],limit_y[0]])
        if x_intersects[1]>=limit_x[0] and x_intersects[1]<=limit_x[1]:
            box_intersection.append([x_intersects[1],limit_y[1]])
        
        t_box = []


        for i in box_intersection:
            if(np.round(self.new_vector[0],6) == 0):
                t_box.append((i[1]-self.new_Point[1])/self.new_vector[1])
            elif(np.round(self.new_vector[1],6) == 0):
                t_box.append((i[0]-self.new_Point[0])/self.new_vector[0])
            else:
                t_box.append((i[0]-self.new_Point[0])/self.new_vector[0])

        if t_box[0]>t_box[1]:
            box_intersection[0],box_intersection[1] = box_intersection[1],box_intersection[0]
        # print("t_box",t_box)

        return np.array(box_intersection)
    
    def to_dict(self):
        # Convert the VortexFilament object to a dictionary for storage
        dict = {
            'Gamma': self.Strength,
            'Point': self.Point.tolist(),  # Convert numpy arrays to lists
            'vec': self.vector.tolist()  # Convert numpy arrays to lists
        }
        if self.start is not None:
            dict['start'] = self.start.tolist()
        else:
            dict['start'] = None
        if self.end is not None:
            dict['end'] = self.end.tolist()
        else:
            dict['end'] = None
        if self.prop is not None:
            dict['prop'] = self.prop
        else:
            dict['prop'] = None
        dict['children'] = []

        for i in self.children:
            dict['children'].append(i.to_dict())
        return dict

    @staticmethod
    def from_dict(data):
        # Recreate the VortexFilament object from a dictionary
        vec = np.array(data['vec'])
        obj = VortexFilament(data['Gamma'], data['Point'], vec, data['start'], data['end'], data['prop'])
        for i in data['children']:
            obj.children.append(VortexFilament.from_dict(i))
        return obj
        

class VortexFilamentCollection:
    def __init__(self):
        self.filaments = []
    def add_filament(self, strength, point, vector, start=None, end=None):
        self.filaments.append(VortexFilament(strength, point, vector, start, end))

def vortfil():
    return html.Div(children=[
    
    #### ============== ####
    #### BUILDING BLOCK ####
    #### ============== ####
    html.H1("Vortex Filaments"),

    dcc.Markdown('''
    \\[...\\] \n
    Helmholtz's vortex theorems: \n
    The circulation strength $\\Gamma$ remains constant along the filament \n
    a vortex filament cannot end in the flow, but: \n
    extends to infinity\n
    ends at a boundary\n
    forms a closed loop\n

    Bior Savart Law: \n

    $$
    d\\vec{V} = \\frac{\\Gamma}{4\\pi} \\frac{d\\vec{l} \\times \\vec{r}}{|\\vec{r}|^3}
    $$
    For a straight filament, the velocity field is given by: \n
    $$
    \\vec{V} = \\frac{\\Gamma}{4\\pi} \\int_{A}^{B} \\frac{sin(\\theta)}{r^2} d\\vec{l}
    $$
    ''',mathjax=True),
    
    html.H2("Vortex Filament tool"),
    dcc.Markdown('''
                 ''',mathjax=True),    
    html.Div(
        style={'display': 'flex', 'justifyContent': 'center', 'gap': '10px', 'alignItems': 'center', 'marginTop': '20px'},
        children=[
            html.Label('x:', style={'marginRight': '5px'}),
            dcc.Input(id='x_point_intercept', type='number', step=0.1, value=0),
            html.Label('y:', style={'marginLeft': '20px', 'marginRight': '5px'}),
            dcc.Input(id='y_point_intercept', type='number', step=0.1, value=0)
        ]),

    html.Br(),

    dcc.Store(id='filam-store'),  # Hidden store for Filam object
    
    html.Label('Strength Slider:'),
    dcc.Slider(-2, 2,
               value=1,
               id='VortexStrength',
              ),

    html.Label('Angle Slider:    '),
    dcc.Slider(-180, 180,
                value=0,
                id='VortexAngle',
                ),

    html.Button('Draw', id='draw-button', n_clicks=0),

    ## Graph updated via app.callable() in main.py
    dcc.Graph(id='vort', config={'clickmode': 'event+select'}),  # Enable clickmode to select points
    
    html.Div(id='selected-point-output'),  # Output the selected point here
    html.Label('First split angle:'),
    dcc.Slider(-180, 180,
               value=1,
               id='angle_1',
              ),

    html.Label('Angle Slider:    '),
    dcc.Slider(-180, 180,
                value=0,
                id='angle_2',
                ),
    html.Button('Split', id='split-button', n_clicks=0),




    #### ============= ####
    #### APPLICATION 1 ####
    #### ============= ####
    html.Hr(),
    html.H2("Application: Source + Uniform"),

    dcc.Markdown('''
    bla bla bla bla bla, velocity graph 2
    ''',mathjax=True),
    

    html.Label('Source Strength Slider:'),
    dcc.Slider(0, 2,
               value=1,
               id='sourceStrength2',
              ),
    html.Label('Freestream Velocity Slider:'),
    dcc.Slider(0.1, 2,
               value=1,
               id='VelInfMagSourceUniform',
              ),

    ## Graph updated via app.callable() in main.py
    dbc.Row([
        dbc.Col([
            dcc.Graph(id='sourceuniformV', mathjax=True),
        ], width=6)
    ], justify='center'),
    ## Graph updated via app.callable() in main.py
    dbc.Row([
        dbc.Col([
            dcc.Graph(id='sourceuniformPS', mathjax=True),
        ], width=4)
    ], justify='center'),


    dcc.Markdown('''
    We can also look at quantities over the body contour...
    ''',mathjax=True),

    ## Graph updated via app.callable() in main.py
    dbc.Row([
        dbc.Col([
            dcc.Graph(id='sourceuniformVelS', mathjax=True),
        ], width=6)
    ], justify='center'),
    ## Graph updated via app.callable() in main.py
    dbc.Row([
        dbc.Col([
            dcc.Graph(id='sourceuniformCpS', mathjax=True),
        ], width=6)
    ], justify='center')
    ])

if __name__ == '__main__':

    # test_wing = VortexFilament(1,[1,1,0],[0.,1,0])
    # test_wing.bend([1,3,0],90)

    # test_wing.draw_vortex_family_3d([0,4],[0,4],[-2,2])
    # X, Y, Z = test_wing.gather_points([0,4],[0,4],[-2,2])

    # fig = test_wing.draw_all([0,4],[0,4],[-2,2])
    # fig.show()

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(X,Y,Z)
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    # plt.show()


    test_wing = VortexFilament(0.5,[-2,1,0],[0,1,0])
    test_wing.bend([-2,2,0],-90)

    wing_for = test_wing.children[0]

    x_splits = np.linspace(-2, 0, 5)[1:-1]
    y = 2
    z = 0
    
    for i in x_splits:
        wing_for.split([1.05,-0.05],[i,y,z],[0,-90])
        wing_for = wing_for.children[0]
    
    x_after_splits = np.linspace(0, 2, 5)[:-1]
    for i in x_after_splits:
        wing_for.split([0.95,0.05],[i,y,z],[0,-90])
        wing_for = wing_for.children[0]
    wing_for.bend([2,y,0],-90)

    test_wing.draw_vortex_family([-3,3],[-3,3])

    plt.xlim([-3,3])
    plt.ylim([-3,3])
    plt.show()

    fig = test_wing.draw_all([-3,3],[-1,5],[-3,3], y_val = 3)
    fig.show()





    print(np.dot(transform,np.array([0,1,0])))
    
    print(np.dot(transform,np.array([0,1,0])))



    test = VortexFilament(10,[1,1,0],[0.,1,0])
    test.split([0.5, 0.5],[1,3,0],[-90,90])
    test.children[1].bend([0.5,3,0],90)
    test.children[0].bend([2,3,0],-90)

    fig = test.draw_all([0,4],[0,4],[-2,2], y_val = 1)
    fig.show()
    # test = VortexFilament(1,[2,1,0],[0.,1,0])
    # test.bend([2,3,0],90)
    # test.children[0].bend([3,3,0],90)

    test.draw_vortex_family([0,4],[0,4])
    plt.xlim([0,4])
    plt.ylim([0,4])
    plt.show()

    x_arr = np.linspace(0,4,100)
    z_arr = np.linspace(-2,2,100)
    X, Z = np.meshgrid(x_arr, z_arr)
    vel = np.zeros(X.shape)
    for i in range(100):
        for j in range(100):
            coord = np.array([X[i,j],1,Z[i,j]])
            vel[i,j] = np.linalg.norm(test.calculate_velocity_family(coord))
    plt.contourf(X,Z,vel, levels = 1000)
    plt.colorbar()
    plt.show()

    X, y, Z, vel = test.get_surface([0,4],[0,4], [-2,2],1,1,num_points=100)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    vel = np.clip(vel,0,1)
    colors = plt.cm.viridis(vel)

    surf = ax.plot_surface(X, y*np.ones(X.shape), Z,facecolors = colors)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    mappable = plt.cm.ScalarMappable(cmap='viridis')
    mappable.set_array(vel)
    fig.colorbar(mappable, ax=ax, shrink=0.5, aspect=5, label='Velocity (vel)')

    plt.show()

    plt.grid()
    plt.axis('scaled')
    plt.gca().set_aspect('equal', 'box')
    plt.xlim([0,4])
    plt.ylim([0,4])
    plt.show()
    # test.end = [1,1,0]
    dictionario = test.to_dict()
    print(dictionario)
    test = VortexFilament.from_dict(dictionario)
    test.draw_vortex_family([0,4],[0,4])
    plt.show()
    
    

    # roll_angle = -np.deg2rad(60)
    # pitch_angle = -np.deg2rad(20)
    # roll_angle2 = np.deg2rad(0)

    # roll_transform = np.array([[1,0,0],[0,np.cos(roll_angle),-np.sin(roll_angle)],[0,np.sin(roll_angle),np.cos(roll_angle)]])
    # pitch_transform = np.array([[np.cos(pitch_angle),0,np.sin(pitch_angle)],[0,1,0],[-np.sin(pitch_angle),0,np.cos(pitch_angle)]])

    # transform = np.dot(pitch_transform, roll_transform)

    # roll_transform2 = np.array([[1,0,0],[0,np.cos(roll_angle2),-np.sin(roll_angle2)],[0,np.sin(roll_angle2),np.cos(roll_angle2)]])
    # transform = np.dot(roll_transform2,transform)

    # transform_test = transform_sp.Rotation.from_euler('xyz', [np.rad2deg(roll_angle),np.rad2deg(pitch_angle),0], degrees=True)
    # print(transform)
    # print(transform_test.as_matrix())

    # print(np.dot(transform,np.array([0,1,0])))