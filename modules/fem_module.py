import fenics as fe
import numpy as np
import mshr as mesher
import sympy 
import os
import re
from scipy.optimize import least_squares as lsq


from image_manipulation_module import ImageManipulation

class FEMSimulation():

#==================================================================
    def __init__(self, storage_dir = None):

        #self.mode = 'test'
        self.use_hg   = False
        self.fast_run = None
        self.mode = 'exp'
        self.experimental_z_n = 1
        self.dimension = 2
        self.image_manipulation_obj = ImageManipulation()
        self.initial_time = 0.0
        self.final_time   = 4.0

        self.diffusion_matrix = None
        self.diffusion_coefficient = 1.0
        self.lam          = 1

        self.dt           = 0.05
        self.error_list   = []
        self.u_exact_str  = None
        self.rhs_fun_str  = None
        self.alpha        = 0
        self.beta         = 0
        self.kappa        = 0
        self.boundary_fun = None
        self.boundary_conditions = None
        self.mesh         = None
        self.u            = None
        self.u_n          = None
        self.rhs_fun      = None
        self.ic_fun       = None
        self.bilinear_form= None
        self.rhs          = None
        self.current_time = 0
        self.function_space = None

        self.fem_opt_data_source_dir  = None
        self.fem_opt_exp_data_vec_dir = None
        self.fem_opt_img_storage_dir  = None
        self.u_experimental = None

        self.experimental_data_vec = []
        self.experimental_data_day = []
        self.experimental_data_sorted_indices = None


        #self.set_parameters()
        self.set_data_dirs()

#==================================================================
    def get_real_time(self):

        return self.current_time + 1

#==================================================================
    def set_data_dirs(self):


        with_hg = 'HG-NPs_tumor-bearing'
        no_hg   = 'NPS_TUMOR-BEARING'

        if self.use_hg:
            hg_state = with_hg

        else:
            hg_state = no_hg

        lambda_label = str(self.lam != 0)

        self.fem_opt_data_source_dir  = os.path.join(os.getcwd(),\
                'raw_data/Tumor-bearing',\
                hg_state)

        self.fem_opt_exp_data_vec_dir = os.path.join(\
                self.fem_opt_data_source_dir,\
                'fem_m_' + str(self.experimental_z_n))

        if not os.path.exists(self.fem_opt_exp_data_vec_dir):
            os.makedirs(self.fem_opt_exp_data_vec_dir)

        txt = 'diffusion_' + 'lambda_' + lambda_label
        self.fem_opt_img_storage_dir = os.path.join(\
                self.fem_opt_data_source_dir,\
                txt)

        if not os.path.exists(self.fem_opt_img_storage_dir):
            os.makedirs(self.fem_opt_img_storage_dir)

        txt = 'solution.pvd'
        fname = os.path.join(self.fem_opt_img_storage_dir, txt)
        self.vtkfile = fe.File(fname)

#==================================================================
    def set_parameters(self):

        if self.mode == 'test': 
            self.alpha        = 3.0
            self.beta         = 1.2
            self.lam          = 1.0
            self.diffusion_matrix = \
                    self.diffusion_coefficient *\
                    np.array([[3., 2.],[1.6125, 1.]])

            print('Lambda: ', self.lam)
            print('Alpha : ', self.alpha)
            print('Beta  : ', self.beta)

        else:
            self.alpha        = 1.8
            self.beta         = 0
            self.lam          = 0.0
            self.kappa        = 0.5
            self.diffusion_matrix =\
                    self.diffusion_coefficient *\
                    np.array([[1., 0.],[0., 1.]])

        if self.dimension == 1:
            self.diffusion_matrix = self.diffusion_coefficient

#==================================================================
    def create_initial_condition_function(self):

        if self.mode == 'test':
            return

        x,y,a,b,k = sympy.symbols('x[0], x[1], alpha, beta, kappa')

        if self.dimension == 1: 
            ic = sympy.exp(-k * ((x-a)**2))

        if self.dimension == 2: 
            ic = sympy.exp(-k * ((x-a)**2 + (y-b)**2))

        ic_str = sympy.printing.ccode(ic)

        self.ic_fun =\
                fe.Expression(ic_str, degree=2,\
                alpha = self.alpha, beta = self.beta, kappa = self.kappa)

#==================================================================
    def create_exact_solution_and_rhs_fun_strings(self):

        if self.mode != 'test':
            return

        print('Creating exact solution and rhs strings')

        x,y,a,b,l,t = sympy.symbols('x[0], x[1], alpha, beta, lam, t')

        if self.dimension == 2: 
            u_exact = 1 + x**2 + a * y**2 + b * t

        if self.dimension == 1: 
            u_exact = 1 + a * x**2 + b * t

        u_t = u_exact.diff(t)

        if self.dimension == 1: 
            grad_u = u_exact.diff(x)
            diffusion_grad_u = self.diffusion_matrix * grad_u
            diffusion_term = diffusion_grad_u.diff(x)

        if self.dimension == 2: 
            grad_u = sympy.Matrix([u_exact]).jacobian([x,y]).T
            diffusion_grad_u = self.diffusion_matrix * grad_u
            diffusion_term = diffusion_grad_u.jacobian([x,y]).trace()

        rhs_fun = u_t - diffusion_term + l*u_exact

        self.u_exact_str = sympy.printing.ccode(u_exact)
        self.rhs_fun_str = sympy.printing.ccode(rhs_fun)

#==================================================================
    def create_rhs_fun(self):

        if self.mode == 'test': 

            print('Creating rhs function')
            self.rhs_fun = fe.Expression(self.rhs_fun_str, degree=2,\
                    alpha = self.alpha,\
                    beta  = self.beta,\
                    lam   = self.lam,\
                    t     = 0)
        else:
            '''
            Zero RHS for the experimental case
            '''
            self.rhs_fun = fe.Constant(0)

#==================================================================
    def create_boundary_conditions(self):

        if self.mode == 'test':

            print('Creating boundary function')
            self.boundary_fun = fe.Expression(self.u_exact_str, degree=2,\
                    alpha = self.alpha, beta = self.beta, t = 0)

        else:
            '''
            Homogeneous boundary conditions
            '''
            self.boundary_fun = fe.Constant(0)


        def is_on_the_boundary(x, on_boundary):
            return on_boundary

        self.boundary_conditions = fe.DirichletBC(\
                self.function_space, self.boundary_fun, is_on_the_boundary)

#==================================================================
    def create_simple_mesh(self):

        #domain      = mesher.Circle(fe.Point(0,0), 1)
        #mesh        = mesher.generate_mesh(domain, 64)
        '''
        domain_vertices = [\
                fe.Point(0.0, 0.0),\
                fe.Point(10.0, 0.0),\
                fe.Point(10.0, 2.0),\
                fe.Point(8.0, 2.0),\
                fe.Point(7.5, 1.0),\
                fe.Point(2.5, 1.0),\
                fe.Point(2.0, 4.0),\
                fe.Point(0.0, 4.0),\
                fe.Point(0.0, 0.0)]

        geo = mesher.Polygon(domain_vertices)
        self.mesh   = mesher.generate_mesh(geo, 64);
        '''
        print('Creating simple mesh')
        nx = ny = 8

        if self.dimension == 1: 
            self.mesh = fe.UnitIntervalMesh(nx)
            #self.mesh = fe.IntervalMesh(nx,-4, 4)


        if self.dimension == 2: 
            self.mesh = fe.UnitSquareMesh(nx, ny)

        '''
        finite_element         = self.function_space.element()
        map_cell_index_to_dofs = self.function_space.dofmap()
        for cell in fe.cells(self.mesh):
            print(map_cell_index_to_dofs.cell_dofs(cell.index()))
            print(finite_element.tabulate_dof_coordinates(cell))
            print('------------')
            break
        '''

#==================================================================
    def create_mesh(self):

        if self.mode == 'test':
            self.create_simple_mesh()
        else:
            self.create_coronal_section_mesh(self.experimental_z_n)

#==================================================================
    def create_coronal_section_mesh(self, experimental_z_n = 0):


        brain_slice = self.image_manipulation_obj.\
                get_brain_slice_from_experimental_z_n(experimental_z_n)

                #get_brain_slice_from_experimental_z_mm(0)

        x_size = 65
        points = brain_slice.\
                generate_boundary_points_ccw(x_size)
        domain_vertices = []

        for x,y in zip(points[0], points[1]):
            domain_vertices.append(fe.Point(x,y))

        geo         = mesher.Polygon(domain_vertices)
        self.mesh   = mesher.generate_mesh(geo, 64);

#==================================================================
    def set_function_spaces(self):

        self.function_space = fe.FunctionSpace(self.mesh, 'P', 1)

#==================================================================
    def compute_error(self):

        if self.mode != 'test':
            return

        error_L2 = fe.errornorm(self.boundary_fun, self.u_n, 'L2')
        error_LI = np.abs(\
                fe.interpolate(\
                self.boundary_fun,self.function_space).vector().get_local() -\
                self.u_n.vector().get_local()\
                ).max()

        print('L2 error at t = {:.3f}: {:.2e}'.format(\
                self.current_time, error_L2))

        print('LI error at t = {:.3f}: {:.2e}'.format(\
                self.current_time, error_LI))

        self.error_list.append(error_L2) 

#==================================================================
    def set_opt_initial_conditions(self):

        self.current_time = self.initial_time
        self.u   = fe.Function(self.function_space)
        self.u_n = fe.Function(self.function_space)
        self.u_n.assign(self.u_experimental[0])
        self.save_snapshot()

#==================================================================
    def set_initial_conditions(self):

        self.current_time = self.initial_time

        #Initial condition
        #self.u_n = fe.project(self.boundary_fun, self.function_space)

        if self.mode == 'test':
            print('Setting initial conditions')
            self.boundary_fun.t = self.current_time
            self.u_n = fe.interpolate(self.boundary_fun, self.function_space)

        else:
            self.u_n = fe.project(self.ic_fun, self.function_space)

        self.u = fe.Function(self.function_space)

        self.compute_error()
        self.save_snapshot()

#==================================================================
    def load_coronal_section_vectors(self):
        
        day_regexp = re.compile(r'vector_day_(?P<day>[0-9]+)')
        for (dir_path, dir_names, file_names) in\
                os.walk(self.fem_opt_exp_data_vec_dir):
            for f in file_names:

                obj = day_regexp.search(f)

                if obj is None:
                    continue

                day_str = obj.group('day')
                day     = int(day_str)

                fname = os.path.join(dir_path, f)

                self.experimental_data_vec.append(np.loadtxt(fname))
                self.experimental_data_day.append(day)

        self.experimental_data_sorted_indices =\
                np.argsort(self.experimental_data_day)

#==================================================================
    def create_coronal_section_vectors(self):

        z_slice_n_regex = re.compile(r'm_?(?P<z_n>[0-9]+)')
        day_folder_regex = re.compile(r'(?P<day>[0-9]+)[Dd]')

        for (dir_path, dir_names, file_names) in\
                os.walk(self.fem_opt_data_source_dir):

            experimental_basename = os.path.basename(dir_path)

            obj = day_folder_regex.match(experimental_basename)

            if obj is None:
                continue

            day = obj.group('day')

            day_dir = os.path.join(self.fem_opt_data_source_dir, dir_path)

            for (inner_dir_path, inner_dir_names, inner_file_names) in\
                    os.walk(day_dir):

                experimental_basename = os.path.basename(inner_dir_path)

                obj = z_slice_n_regex.search(experimental_basename)

                if obj is None:
                    continue

                m_value_str = obj.group('z_n')
                m_value = int(m_value_str)

                if m_value != self.experimental_z_n: 
                    continue

                local_source_dir = os.path.join(day_dir, inner_dir_path)

                fname = 'vector_day_' + day + '.txt'
                storage_fname = os.path.join(\
                        self.fem_opt_exp_data_vec_dir,\
                        fname)

                self.brain_slice_data_to_array(\
                        m_value,\
                        local_source_dir,\
                        storage_fname)


#==================================================================
    def brain_slice_data_to_array(self,\
            experimental_z_n,\
            source_dir,\
            storage_fname = None):

        self.create_coronal_section_mesh(experimental_z_n)

        #Only for linear elements
        self.set_function_spaces()

        dof_to_vertex_vec = fe.dof_to_vertex_map(self.function_space)
        coordinates = self.mesh.coordinates()[dof_to_vertex_vec]
        x = coordinates[:,0]
        y = coordinates[:,1]

        SP = self.image_manipulation_obj.\
                create_slice_properties_object(source_dir)

        u = SP.map_from_model_xy_to_experimental_matrix(x, y)

        if storage_fname is None:
            return u

        np.savetxt(storage_fname, u, fmt='%d')

        return u

#==================================================================
    def brain_slice_data_to_fem_vector(self,\

            experimental_z_n,\
            source_dir,\
            fem_vector,\
            storage_fname = None):

        u = self.brain_slice_data_to_array(\
                experimental_z_n,\
                source_dir,\
                storage_fname)

        fem_vector.vector().set_local(u)

#==================================================================
    def create_opt_bilinear_form_and_rhs(self):


        # Define variational problem
        u = fe.TrialFunction(self.function_space)
        v = fe.TestFunction(self.function_space)

        self.bilinear_form = (1 + self.lam * self.dt) *\
                (u * v * fe.dx) + self.dt * self.diffusion_coefficient *\
                (fe.dot(fe.grad(u), fe.grad(v)) * fe.dx)

        self.rhs = (self.u_n + self.dt * self.rhs_fun) * v * fe.dx

#==================================================================
    def create_bilinear_form_and_rhs(self):


        # Define variational problem
        u = fe.TrialFunction(self.function_space)
        v = fe.TestFunction(self.function_space)

        if self.dimension == 1:
            diffusion_tensor = self.diffusion_matrix
            self.bilinear_form = (1 + self.lam * self.dt) * (u * v * fe.dx) +\
                    self.dt * (fe.dot(\
                    diffusion_tensor * fe.grad(u),\
                    fe.grad(v)) * fe.dx)

        if self.dimension == 2:
            diffusion_tensor = fe.as_matrix(self.diffusion_matrix)
            self.bilinear_form = (1 + self.lam * self.dt) * (u * v * fe.dx) +\
                    self.dt * (fe.dot(\
                    fe.dot(diffusion_tensor, fe.grad(u)),\
                    fe.grad(v)) * fe.dx)


        self.rhs = (self.u_n + self.dt * self.rhs_fun) * v * fe.dx


#==================================================================
    def solve_problem(self):

        fe.solve(self.bilinear_form == self.rhs,\
                self.u, self.boundary_conditions)

#==================================================================
    def save_snapshot(self):

        if self.fast_run:
            return

        self.vtkfile << (self.u_n, self.current_time)

#==================================================================
    def load_opt_experimental_data(self):

        self.load_coronal_section_vectors()
        self.u_experimental = ['0'] * len(self.experimental_data_day)
        for c, idx in enumerate(self.experimental_data_sorted_indices):
            u = fe.Function(self.function_space)
            u.vector().set_local(self.experimental_data_vec[c])
            self.u_experimental[idx] = u

#==================================================================
    def optimization_setup(self):

        self.create_mesh()
        self.set_function_spaces()
        self.create_rhs_fun()
        self.create_boundary_conditions()
        self.load_opt_experimental_data()


#==================================================================
    def optimization_run(self):

        self.set_opt_initial_conditions()
        self.create_opt_bilinear_form_and_rhs()

        while self.dt < 2*np.abs(self.current_time - self.final_time):
            
            self.current_time += self.dt
            print('D = {:0.2e}, L = {:0.2e}, t = {:0.2f}'.format(\
                    self.diffusion_coefficient,\
                    self.lam,\
                    self.get_real_time()))
            self.boundary_fun.t = self.current_time
            self.rhs_fun.t      = self.current_time
            self.solve_problem()
            self.u_n.assign(self.u)
            self.save_snapshot()

        print('Real time:', self.get_real_time(), 'days')
        
#==================================================================
    def optimize(self):

        self.optimization_setup()
        p = np.array([0.02289848, 0.01645834])

        jump_opt = False

        if jump_opt == False:

            self.fast_run = True
            p = np.array([1.])

            if self.lam != 0:
                p = np.array([1., 1.])

            obj = lsq(self.objective_function, p)
            p = obj.x

        self.fast_run = False
        self.objective_function(p)

        print('Plotting complete')

        if jump_opt == False:
            print('Optimal:', obj.x)
        else:
            return

        lambda_label = str(self.lam != 0)
        txt = 'opt_lambda_' + lambda_label + '.txt'

        fname = os.path.join(self.fem_opt_exp_data_vec_dir, txt)
        np.savetxt(fname, p)



#==================================================================
    def objective_function(self,params):

        self.diffusion_coefficient = params[0]

        if 1 < params.size:
            self.lam = params[1]

        self.optimization_run()
        v = self.u.vector() - self.u_experimental[1].vector()

        return v.get_local()
        


#==================================================================
    def run(self):

        self.create_exact_solution_and_rhs_fun_strings()
        self.create_initial_condition_function()
        self.create_rhs_fun()
        self.create_mesh()
        self.set_function_spaces()
        self.create_boundary_conditions()
        self.set_initial_conditions()
        self.create_bilinear_form_and_rhs()

        while self.current_time < self.final_time: 
            
            self.current_time += self.dt
            print('t = {:0.2f}'.format(self.current_time))
            self.boundary_fun.t = self.current_time
            self.rhs_fun.t      = self.current_time
            self.solve_problem()
            self.u_n.assign(self.u)
            self.compute_error()
            self.save_snapshot()

            print('------------------------')
            
        
        print('Alles ist gut')
        print(np.linalg.det(self.diffusion_matrix))
        print(np.linalg.eig(self.diffusion_matrix)[0])
        



