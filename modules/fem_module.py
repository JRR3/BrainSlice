import fenics as fe
import numpy as np
import mshr as mesher
import sympy 


from image_manipulation_module import ImageManipulation

class FEMSimulation():

#==================================================================
    def __init__(self):

        self.mode = 'test'
        #self.mode = 'exp'
        self.image_manipulation_obj = ImageManipulation()
        self.initial_time = 0.0
        self.final_time   = 2.0
        self.diffusion_matrix = None
        self.dt           = 0.05
        self.error_list   = []
        self.u_exact_str  = None
        self.rhs_fun_str  = None
        self.alpha        = 0
        self.beta         = 0
        self.kappa        = 0.5
        self.lam          = 0
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
        self.vtkfile      = fe.File('diffusion/solution.pvd')

        if self.mode == 'test': 
            self.alpha        = 3.0
            self.beta         = 1.2
            self.lam          = 1.0
            self.diffusion_matrix = np.array([[1., 0.],[0., 1.]])

            print('Lambda: ', self.lam)
            print('Alpha : ', self.alpha)
            print('Beta  : ', self.beta)
        else:
            self.alpha        = 1.8
            self.beta         = 0
            self.diffusion_matrix = np.array([[1., 0],[0, 1.]])

#==================================================================
    def create_initial_condition_function(self):

        if self.mode == 'test':
            return

        x,y,a,b,k = sympy.symbols('x[0], x[1], alpha, beta, kappa')
        ic = exp(-kappa * ((x-a)**2 + (y-b)**2))
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
        u_exact = 1 + x**2 + a * y**2 + b * t
        u_t = u_exact.diff(t)
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
            self.rhs_fun = fe.Constant(0)

#==================================================================
    def create_boundary_conditions(self):

        if self.mode == 'test':

            print('Creating boundary function')
            self.boundary_fun = fe.Expression(self.u_exact_str, degree=2,\
                    alpha = self.alpha, beta = self.beta, t = 0)

        else:
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
        nx = ny     = 8
        self.mesh   = fe.UnitSquareMesh(nx, ny)

        coordinates = self.mesh.coordinates()

        for cell in fe.cells(self.mesh):
            #print(map_cell_index_to_dofs.cell_dofs(cell.index()))
            #print(finite_element.tabulate_dof_coordinates(cell))
            #print('------------')
            break

#==================================================================
    def create_mesh(self):

        if self.mode == 'test':
            self.create_simple_mesh()
        else:
            self.create_coronal_section_mesh()

#==================================================================
    def create_coronal_section_mesh(self):

        brain_slice = self.image_manipulation_obj.\
                get_brain_slice_from_experimental_z_mm(0)

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

        self.function_space = fe.FunctionSpace(self.mesh, 'P', 2)

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
    def create_bilinear_form_and_rhs(self):

        diffusion_tensor = fe.as_matrix(self.diffusion_matrix)

        finite_element         = self.function_space.element()
        map_cell_index_to_dofs = self.function_space.dofmap()

        #Only for linear elements
        #dof_to_vertex_vec      = fe.dof_to_vertex_map(self.function_space)

        # Define variational problem
        u = fe.TrialFunction(self.function_space)
        v = fe.TestFunction(self.function_space)

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
        self.vtkfile << (self.u_n, self.current_time)

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

        c = 0

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
            
            c += 1

        
        print('Alles ist gut')



