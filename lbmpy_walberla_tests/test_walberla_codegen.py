import unittest

import sympy as sp

import pystencils as ps
from lbmpy.boundaries import UBB, NoSlip
from lbmpy.creationfunctions import create_lb_method, create_lb_update_rule
from lbmpy_walberla import RefinementScaling, generate_boundary, generate_lattice_model
from lbmpy_walberla.sparse import ListLbGenerator
from pystencils_walberla import generate_pack_info_for_field, generate_pack_info_from_kernel
from pystencils_walberla.cmake_integration import ManualCodeGenerationContext


class WalberlaLbmpyCodegenTest(unittest.TestCase):

    @staticmethod
    def test_lattice_model():
        with ManualCodeGenerationContext() as ctx:
            force_field = ps.fields("force(3): [3D]", layout='fzyx')
            omega = sp.Symbol("omega")

            lb_method = create_lb_method(stencil='D3Q19', method='srt', relaxation_rates=[omega], compressible=True,
                                         force_model='guo', force=force_field.center_vector)

            scaling = RefinementScaling()
            scaling.add_standard_relaxation_rate_scaling(omega)
            scaling.add_force_scaling(force_field)

            generate_lattice_model(ctx, 'SrtWithForceFieldModel', lb_method, refinement_scaling=scaling,
                                   update_rule_params={'compressible': True})
            generate_boundary(ctx, 'MyUBB', UBB([0.05, 0, 0]), lb_method)
            generate_boundary(ctx, 'MyNoSlip', NoSlip(), lb_method)
            assert 'static const bool compressible = true;' in ctx.files['SrtWithForceFieldModel.h']

    @staticmethod
    def test_sparse():
        from lbmpy.creationfunctions import create_lb_collision_rule
        from pystencils import show_code
        g = ListLbGenerator(create_lb_collision_rule())
        kernel_code = str(show_code(g.kernel()))
        assert 'num_cells' in kernel_code
        setter_code = str(show_code(g.setter_ast()))
        assert 'num_cells' in setter_code
        getter_code = str(show_code(g.getter_ast()))
        assert 'num_cells' in getter_code

    @staticmethod
    def test_pack_info():
        with ManualCodeGenerationContext() as ctx:
            f = ps.fields("f(9): [3D]")
            generate_pack_info_for_field(ctx, 'MyPackInfo1', f)

            lb_assignments = create_lb_update_rule(stencil='D3Q19', method='srt').main_assignments
            generate_pack_info_from_kernel(ctx, 'MyPackInfo2', lb_assignments)

    @staticmethod
    def test_incompressible():
        with ManualCodeGenerationContext() as ctx:
            omega = sp.Symbol("omega")

            lb_method = create_lb_method(stencil='D3Q19', method='srt', relaxation_rates=[omega], compressible=False)
            generate_lattice_model(ctx, 'Model', lb_method, update_rule_params={'compressible': False})
            assert 'static const bool compressible = false;' in ctx.files['Model.h']

    @staticmethod
    def test_output_field():
        with ManualCodeGenerationContext(openmp=True, double_accuracy=True) as ctx:
            omega_field = ps.fields("omega_out: [3D]", layout='fzyx')
            parameters = {
                'stencil': 'D3Q19',
                'method': 'trt',
                'smagorinsky': True,
                'omega_output_field': omega_field,
            }
            lb_method = create_lb_method(**parameters)
            generate_lattice_model(ctx, 'Model', lb_method, update_rule_params=parameters)
