import sympy as sp
from sympy.tensor import IndexedBase
from jinja2 import Environment, PackageLoader, Template
import os
import inspect

from pystencils.astnodes import SympyAssignment
from pystencils.assignment_collection.assignment_collection import AssignmentCollection
from pystencils.sympyextensions import get_symmetric_part
from pystencils.field import offset_to_direction_string, Field
from pystencils.backends.cbackend import CustomSympyPrinter, CBackend
from pystencils.data_types import TypedSymbol
from pystencils_walberla.sweep import KernelInfo
from pystencils_walberla.jinja_filters import add_pystencils_filters_to_jinja_env
from pystencils.cpu import add_openmp, create_kernel

from lbmpy.relaxationrates import relaxation_rate_scaling
from lbmpy.creationfunctions import create_lb_method, update_with_default_parameters,\
    create_lb_ast
from lbmpy.updatekernels import create_stream_pull_only_kernel

cppPrinter = CustomSympyPrinter()


REFINEMENT_SCALE_FACTOR = sp.Symbol("levelScaleFactor")


def stencil_switch_statement(stencil, values):
    template = Template("""
    using namespace stencil;
    switch( direction ) {
        {% for directionName, value in dir_to_value_dict.items() -%}
            case {{directionName}}: return {{value}};
        {% endfor -%}
        default:
            WALBERLA_ABORT("Invalid Direction");
    }
    """)

    dir_to_value_dict = {offset_to_direction_string(d): cppPrinter.doprint(v) for d, v in zip(stencil, values)}
    return template.render(dir_to_value_dict=dir_to_value_dict)


def field_and_symbol_substitute(expr, variable_prefix="lm.", variables_without_prefix=[]):
    variables_without_prefix = [v.name if isinstance(v, sp.Symbol) else v for v in variables_without_prefix]
    substitutions = {}
    for sym in expr.atoms(sp.Symbol):
        if isinstance(sym, Field.Access):
            fa = sym
            prefix = "" if fa.field.name in variables_without_prefix else variable_prefix
            if fa.field.index_dimensions == 0:
                substitutions[fa] = sp.Symbol("%s%s->get(x,y,z)" % (prefix, fa.field.name))
            else:
                assert fa.field.index_dimensions == 1, "walberla supports only 0 or 1 index dimensions"
                substitutions[fa] = sp.Symbol("%s%s->get(x,y,z,%s)" % (prefix, fa.field.name, fa.index[0]))
        else:
            if sym.name not in variables_without_prefix:
                substitutions[sym] = sp.Symbol(variable_prefix + sym.name)
    return expr.subs(substitutions)


def expression_to_code(expr, variable_prefix="lm.", variables_without_prefix=[]):
    """
    Takes a sympy expression and creates a C code string from it. Replaces field accesses by
    walberla field accesses i.e. field_W^1 -> field->get(-1, 0, 0, 1)
    :param expr: sympy expression
    :param variable_prefix: all variables (and field) are prefixed with this string
                           this is used for member variables mostly
    :param variables_without_prefix: this variables are not prefixed
    :return: code string
    """
    return cppPrinter.doprint(field_and_symbol_substitute(expr, variable_prefix, variables_without_prefix))


def equations_to_code(equations, variable_prefix="lm.", variables_without_prefix=[]):
    def type_eq(eq):
        return eq.subs({s: TypedSymbol(s.name, "double") for s in eq.atoms(sp.Symbol)})

    if isinstance(equations, AssignmentCollection):
        equations = equations.all_assignments

    variables_without_prefix = list(variables_without_prefix)
    c_backend = CBackend()
    result = []
    left_hand_side_names = [e.lhs.name for e in equations]
    for eq in equations:
        assignment = SympyAssignment(type_eq(eq.lhs),
                                     field_and_symbol_substitute(eq.rhs, variable_prefix,
                                                                 variables_without_prefix + left_hand_side_names))
        result.append(c_backend(assignment))
    return "\n".join(result)


def generate_lattice_model(lattice_model_name=None, optimization={}, refinement_scaling=None,
                           lb_method=None, **kwargs):
    """
    Creates a waLBerla lattice model consisting of a source and header file

    :param lattice_model_name: name of the generated lattice model class. If None, it is assumed that this function was
                             called from a .gen.py file and the filename is taken as lattice model name
    :param optimization: see documentation of create_lb_ast
    :param kwargs: see documentation of create_lb_ast
    :param refinement_scaling: dict from parameter symbol (e.g. relaxationRate, force parameter) to an expression
                              how the parameter scales on refined blocks. The refinement factor is represented by
                              the global symbol REFINEMENT_SCALE_FACTOR
    :return: tuple with code strings (header, sources)
    """
    if lattice_model_name is None:
        script_file_name = inspect.stack()[-1][1]
        if script_file_name.endswith(".cuda.gen.py"):
            raise ValueError("GPU Lattice Model are not yet supported")
        elif script_file_name.endswith(".gen.py"):
            file_name = script_file_name[:-len(".gen.py")]
            lattice_model_name = os.path.split(file_name)[1]
        else:
            raise ValueError("Not called from a .gen.py file and lattice_model_name is missing")

    if 'field_layout' not in optimization:
        # usually a numpy layout is chosen by default i.e. xyzf - which is bad for waLBerla where at least the spatial
        # coordinates should be ordered in reverse direction i.e. zyx
        optimization['field_layout'] = 'fzyx'

    params, opt_params = update_with_default_parameters(kwargs, optimization)

    stencil_name = params['stencil']

    if params['force_model'] != 'none' and params['force'] == (0, 0, 0):
        params['force'] = sp.symbols("force:3")

    params['field_name'] = 'pdfs'
    params['temporary_field_name'] = 'pdfs_tmp'
    if not lb_method:
        lb_method = create_lb_method(**params)
    del params['lb_method']
    stream_collide_ast = create_lb_ast(lb_method=lb_method, optimization=opt_params, **params)
    stream_collide_ast.function_name = 'kernel_streamCollide'

    params['kernel_type'] = 'collide_only'
    collide_ast = create_lb_ast(lb_method=lb_method, optimization=opt_params, **params)
    collide_ast.function_name = 'kernel_collide'

    stream_update_rule = create_stream_pull_only_kernel(lb_method.stencil, src_field_name=params['field_name'],
                                                        dst_field_name=params['temporary_field_name'],
                                                        generic_layout=opt_params['field_layout'])
    stream_ast = create_kernel(stream_update_rule.all_assignments)
    stream_ast.function_name = 'kernel_stream'
    
    add_openmp(stream_ast, num_threads=opt_params['openmp'])
    add_openmp(collide_ast, num_threads=opt_params['openmp'])
    add_openmp(stream_collide_ast, num_threads=opt_params['openmp'])

    vel_symbols = lb_method.conserved_quantity_computation.first_order_moment_symbols
    rho_sym = sp.Symbol("rho")
    pdfs_sym = sp.symbols("f_:%d" % (len(lb_method.stencil),))
    vel_arr_symbols = [IndexedBase(sp.Symbol('u'), shape=(1,))[i] for i in range(len(vel_symbols))]
    momentum_density_symbols = [sp.Symbol("md_%d" % (i,)) for i in range(len(vel_symbols))]

    equilibrium = lb_method.get_equilibrium_terms().subs({a: b for a, b in zip(vel_symbols, vel_arr_symbols)})
    symmetric_equilibrium = get_symmetric_part(equilibrium, vel_arr_symbols)
    asymmetric_equilibrium = sp.expand(equilibrium - symmetric_equilibrium)

    force_model = lb_method.force_model
    macroscopic_velocity_shift = None
    if force_model:
        if hasattr(force_model, 'macroscopic_velocity_shift'):
            macroscopic_velocity_shift = [expression_to_code(e, "lm.", ["rho"])
                                          for e in force_model.macroscopic_velocity_shift(rho_sym)]

    cqc = lb_method.conserved_quantity_computation

    eq_input_from_input_eqs = cqc.equilibrium_input_equations_from_init_values(sp.Symbol("rhoIn"), vel_arr_symbols)
    density_velocity_setter_macroscopic_values = equations_to_code(eq_input_from_input_eqs,
                                                                   variables_without_prefix=['rhoIn', 'u'])
    momentum_density_getter = cqc.output_equations_from_pdfs(pdfs_sym, {'density': rho_sym,
                                                                        'momentum_density': momentum_density_symbols})

    context = {
        'class_name': lattice_model_name,
        'stencil_name': stencil_name,
        'D': lb_method.dim,
        'Q': len(lb_method.stencil),
        'compressible': 'true' if params['compressible'] else 'false',
        'weights': ",".join(str(w.evalf()) for w in lb_method.weights),
        'inverse_weights': ",".join(str((1/w).evalf()) for w in lb_method.weights),

        'equilibrium_order': params['equilibrium_order'],

        'equilibrium_from_direction': stencil_switch_statement(lb_method.stencil, equilibrium),
        'symmetric_equilibrium_from_direction': stencil_switch_statement(lb_method.stencil, symmetric_equilibrium),
        'asymmetric_equilibrium_from_direction': stencil_switch_statement(lb_method.stencil, asymmetric_equilibrium),
        'equilibrium': [cppPrinter.doprint(e) for e in equilibrium],

        'macroscopic_velocity_shift': macroscopic_velocity_shift,
        'density_getters': equations_to_code(cqc.output_equations_from_pdfs(pdfs_sym, {"density": rho_sym}),
                                             variables_without_prefix=[e.name for e in pdfs_sym]),
        'momentum_density_getter': equations_to_code(momentum_density_getter, variables_without_prefix=pdfs_sym),
        'density_velocity_setter_macroscopic_values': density_velocity_setter_macroscopic_values,

        'refinement_scaling': refinement_scaling,

        'stream_collide_kernel': KernelInfo(stream_collide_ast, ['pdfs_tmp'], [('pdfs', 'pdfs_tmp')]),
        'collide_kernel': KernelInfo(collide_ast, [], []),
        'stream_kernel': KernelInfo(stream_ast, ['pdfs_tmp'], [('pdfs', 'pdfs_tmp')]),
        'target': 'cpu',
        'namespace': 'lbm',
    }

    env = Environment(loader=PackageLoader('lbmpy_walberla'))
    add_pystencils_filters_to_jinja_env(env)

    header_file = env.get_template('LatticeModel.tmpl.h').render(**context)
    cpp_file = env.get_template('LatticeModel.tmpl.cpp').render(**context)
    return header_file, cpp_file, context


def generate_lattice_model_files(class_name, *args, **kwargs):
    """
    :param kwargs: see documentation of create_lb_ast, additionally
                   an instance of RefinementScaling can be passed with the 'refinementScaling' keyword
    """
    from pystencils_walberla.cmake_integration import codegen

    def generate_lm():
        header, sources, _ = generate_lattice_model(class_name, *args, **kwargs)
        return header, sources

    codegen.register([class_name + ".h", class_name + ".cpp"], generate_lm)


class RefinementScaling:
    levelScaleFactor = sp.Symbol("levelScaleFactor")

    def __init__(self):
        self.scaling_info = []

    def add_standard_relaxation_rate_scaling(self, viscosity_relaxation_rate):
        self.add_scaling(viscosity_relaxation_rate, relaxation_rate_scaling)

    def add_force_scaling(self, force_parameter):
        self.add_scaling(force_parameter, lambda param, factor: param * factor)

    def add_scaling(self, parameter, scaling_rule):
        """
        Adds a scaling rule, how parameters on refined blocks are modified

        :param parameter: parameter to modify: may either be a Field, Field.Access or a Symbol
        :param scaling_rule: function taking the parameter to be scaled as symbol and the scaling factor i.e.
                            how much finer the current block is compared to coarsest resolution
        """
        if isinstance(parameter, Field):
            field = parameter
            name = field.name
            if field.index_dimensions > 0:
                scaling_type = 'field_with_f'
                field_access = field(sp.Symbol("f"))
            else:
                scaling_type = 'field_xyz'
                field_access = field.center
            expr = scaling_rule(field_access, self.levelScaleFactor)
            self.scaling_info.append((scaling_type, name, expression_to_code(expr, '')))
        elif isinstance(parameter, Field.Access):
            field_access = parameter
            expr = scaling_rule(field_access, self.levelScaleFactor)
            name = field_access.field.name
            self.scaling_info.append(('field_xyz', name, expression_to_code(expr, '')))
        elif isinstance(parameter, sp.Symbol):
            expr = scaling_rule(parameter, self.levelScaleFactor)
            self.scaling_info.append(('normal', parameter.name, expression_to_code(expr, '')))
        elif isinstance(parameter, list) or isinstance(parameter, tuple):
            for p in parameter:
                self.add_scaling(p, scaling_rule)
        else:
            raise ValueError("Invalid value for viscosityRelaxationRate")
