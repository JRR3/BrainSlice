import fenics as fe
import numpy as np
import mshr as mesher
import sympy as sym 

initial_time = 0.0
final_time   = 2.0
diffusion_matrix = np.array([[2.,1.],[1.,4.]])
diffusion_tensor = fe.as_matrix(diffusion_matrix)
dt               = 0.05

#domain      = mesher.Circle(fe.Point(0,0), 1)
#mesh        = mesher.generate_mesh(domain, 64)
nx = ny     = 8
mesh        = fe.UnitSquareMesh(nx, ny)
error_list  = []
coordinates = mesh.coordinates()

#Function space
V = fe.FunctionSpace(mesh, 'P', 2)

finite_element         = V.element()
map_cell_index_to_dofs = V.dofmap()

#Only for linear elements
#dof_to_vertex_vec      = fe.dof_to_vertex_map(V)


for cell in fe.cells(mesh):
    #print(map_cell_index_to_dofs.cell_dofs(cell.index()))
    #print(finite_element.tabulate_dof_coordinates(cell))
    #print('------------')
    pass

x,y,a,b,l,t = sym.symbols('x[0], x[1], alpha, beta, lam, t')
u_exact = 1 + x**2 + a * y**2 + b * t
u_t = u_exact.diff(t)
grad_u = sym.Matrix([u_exact]).jacobian([x,y]).T
diffusion_grad_u = diffusion_matrix * grad_u
diffusion_term = diffusion_grad_u.jacobian([x,y]).trace()
rhs_fun = u_t - diffusion_term + l*u_exact
u_exact = sym.printing.ccode(u_exact)
rhs_fun = sym.printing.ccode(rhs_fun)
print(u_exact)
print(rhs_fun)

alpha        = 3.0
beta         = 1.2
lam          = 1.0


rhs_fun = fe.Expression(rhs_fun, degree=2,\
        alpha = alpha, beta = beta, lam = lam, t = 0)

#Boundary condition
u_boundary = fe.Expression(u_exact, degree=2,\
        alpha = alpha, beta = beta, t = 0)

def is_on_the_boundary(x, on_boundary):
    return on_boundary

boundary_conditions = fe.DirichletBC(V, u_boundary, is_on_the_boundary)

# Define variational problem
u = fe.TrialFunction(V)
v = fe.TestFunction(V)


current_time = initial_time
u_boundary.t = current_time

#Initial condition
#u_n = fe.project(u_boundary, V)

u_n = fe.interpolate(u_boundary, V)

bilinear_form = (1 + lam * dt) * (u * v * fe.dx) +\
        dt * (fe.dot(\
        fe.dot(diffusion_tensor, fe.grad(u)),\
        fe.grad(v)) * fe.dx)

u   = fe.Function(V)

rhs = (u_n + dt * rhs_fun) * v * fe.dx

error_L2 = fe.errornorm(u_boundary, u_n, 'L2')
error_LI = np.abs(\
        fe.interpolate(\
        u_boundary,V).vector().get_local() -\
        u_n.vector().get_local()\
        ).max()
print('L2 error at t = {:.3f}: {:.2e}'.format(current_time, error_L2))
print('LI error at t = {:.3f}: {:.2e}'.format(current_time, error_LI))
error_list.append(error_L2) 

vtkfile = fe.File('diffusion/solution.pvd')
vtkfile << (u_n, current_time)


while current_time < final_time: 
    
    current_time += dt
    u_boundary.t = current_time
    rhs_fun.t    = current_time

    fe.solve(bilinear_form == rhs, u, boundary_conditions)

    # Compute errors
    error_L2 = fe.errornorm(u_boundary, u_n, 'L2')
    error_LI = np.abs(\
            fe.interpolate(\
            u_boundary,V).vector().get_local() -\
            u.vector().get_local()\
            ).max()

    print('L2 error at t = {:.3f}: {:.2e}'.format(current_time, error_L2))
    print('LI error at t = {:.3f}: {:.2e}'.format(current_time, error_LI))
    error_list.append(error_L2) 

    u_n.assign(u)

    # Save solution to file in VTK format
    vtkfile << (u_n, current_time)

    print('------------------------')

#print(np.array(error_list))

#Compute maximum error at vertices
#vertex_values_u_D = u_D.compute_vertex_values(mesh)
#vertex_values_u   = u.compute_vertex_values(mesh)
#error_max         = np.max(np.abs(vertex_values_u_D - vertex_values_u))

