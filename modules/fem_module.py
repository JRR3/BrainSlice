import fenics as fe
import numpy as np
import mshr as mesher
import sympy 
#import dolfin 

class FEMSimulation():

#==================================================================
    def __init__(self):

        self.initial_time = 0.0
        self.final_time   = 2.0
        self.diffusion_matrix = np.array([[2.,1.],[1.,4.]])
        self.dt           = 0.05
        self.error_list   = []
        self.u_exact_str  = None
        self.rhs_fun_str  = None
        self.alpha        = 3.0
        self.beta         = 1.2
        self.lam          = 1.0
        self.u_boundary   = None
        self.boundary_conditions = None
        self.mesh         = None
        self.u            = None
        self.u_n          = None
        self.rhs_fun      = None
        self.bilinear_form= None
        self.rhs          = None
        self.current_time = 0
        self.function_space = None
        self.vtkfile      = fe.File('diffusion/solution.pvd')

#==================================================================
    def create_exact_solution_and_rhs_fun_strings(self):
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

        self.rhs_fun = fe.Expression(self.rhs_fun_str, degree=2,\
                alpha = self.alpha, beta = self.beta, lam = self.lam, t = 0)

#==================================================================
    def create_boundary_conditions(self):

        self.u_boundary = fe.Expression(self.u_exact_str, degree=2,\
                alpha = self.alpha, beta = self.beta, t = 0)

        def is_on_the_boundary(x, on_boundary):
            return on_boundary

        self.boundary_conditions = fe.DirichletBC(\
                self.function_space, self.u_boundary, is_on_the_boundary)

#==================================================================
    def create_mesh(self):

        #domain      = mesher.Circle(fe.Point(0,0), 1)
        #mesh        = mesher.generate_mesh(domain, 64)
        nx = ny     = 8
        self.mesh   = fe.UnitSquareMesh(nx, ny)
        M = fe.Mesh()
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

        M = mesher.generate_mesh(domain_vertices, 64);
        exit()

        coordinates = self.mesh.coordinates()

        for cell in fe.cells(self.mesh):
            #print(map_cell_index_to_dofs.cell_dofs(cell.index()))
            #print(finite_element.tabulate_dof_coordinates(cell))
            #print('------------')
            break

#==================================================================
    def set_function_spaces(self):

        self.function_space = fe.FunctionSpace(self.mesh, 'P', 2)

#==================================================================
    def compute_error(self):

        error_L2 = fe.errornorm(self.u_boundary, self.u_n, 'L2')
        error_LI = np.abs(\
                fe.interpolate(\
                self.u_boundary,self.function_space).vector().get_local() -\
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
        self.u_boundary.t = self.current_time

        #Initial condition
        #self.u_n = fe.project(self.u_boundary, self.function_space)

        self.u_n = fe.interpolate(self.u_boundary, self.function_space)
        self.u   = fe.Function(self.function_space)

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
        self.create_rhs_fun()
        self.create_mesh()
        self.set_function_spaces()
        self.create_boundary_conditions()
        self.set_initial_conditions()
        self.create_bilinear_form_and_rhs()

        while self.current_time < self.final_time: 
            
            self.current_time += self.dt
            self.u_boundary.t = self.current_time
            self.rhs_fun.t    = self.current_time
            self.solve_problem()
            self.u_n.assign(self.u)
            self.compute_error()
            self.save_snapshot()

            print('------------------------')
        
        print('Alles ist gut')



