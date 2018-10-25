import numpy as np
from jinja2 import Environment, PackageLoader

from pystencils import Field, FieldType
from pystencils.data_types import create_type, TypedSymbol
from pystencils_walberla.jinja_filters import add_pystencils_filters_to_jinja_env

from lbmpy.boundaries.boundaryhandling import create_lattice_boltzmann_boundary_kernel
from pystencils.boundaries.createindexlist import numpy_data_type_for_boundary_object, \
    boundary_index_array_coordinate_names, direction_member_name
from lbmpy_walberla.walberla_lbm_generation import KernelInfo


def struct_from_numpy_dtype(struct_name, numpy_dtype):
    result = "struct %s { \n" % (struct_name,)

    equality_compare = []
    constructor_params = []
    constructor_initializer_list = []
    for name, (sub_type, offset) in numpy_dtype.fields.items():
        pystencils_type = create_type(sub_type)
        result += "    %s %s;\n" % (pystencils_type, name)
        if name in boundary_index_array_coordinate_names or name == direction_member_name:
            constructor_params.append("%s %s_" % (pystencils_type, name))
            constructor_initializer_list.append("%s(%s_)" % (name, name))
        else:
            constructor_initializer_list.append("%s()" % name)
        if pystencils_type.is_float():
            equality_compare.append("floatIsEqual(%s, o.%s)" % (name, name))
        else:
            equality_compare.append("%s == o.%s" % (name, name))

    result += "    %s(%s) : %s {}\n" % \
              (struct_name, ", ".join(constructor_params), ", ".join(constructor_initializer_list))
    result += "    bool operator==(const %s & o) const {\n        return %s;\n    }\n" % \
              (struct_name, " && ".join(equality_compare))
    result += "};\n"
    return result


def create_boundary_class(boundary_object, lb_method, double_precision=True, target='cpu'):
    struct_name = "IndexInfo"
    index_struct_dtype = numpy_data_type_for_boundary_object(boundary_object, lb_method.dim)

    pdf_field = Field.create_generic('pdfs', lb_method.dim, np.float64 if double_precision else np.float32,
                                     index_dimensions=1, layout='fzyx', index_shape=[len(lb_method.stencil)])

    index_field = Field('indexVector', FieldType.INDEXED, index_struct_dtype, layout=[0],
                        shape=(TypedSymbol("indexVectorSize", create_type(np.int64)), 1), strides=(1, 1))

    kernel = create_lattice_boltzmann_boundary_kernel(pdf_field, index_field, lb_method, boundary_object, target=target)

    stencil_info = [(i, ", ".join([str(e) for e in d])) for i, d in enumerate(lb_method.stencil)]

    context = {
        'class_name': boundary_object.name,
        'StructName': struct_name,
        'StructDeclaration': struct_from_numpy_dtype(struct_name, index_struct_dtype),
        'kernel': KernelInfo(kernel, [], [], []),
        'stencil_info': stencil_info,
        'dim': lb_method.dim,
        'target': target,
        'namespace': 'lbm',
    }

    env = Environment(loader=PackageLoader('lbmpy_walberla'))
    add_pystencils_filters_to_jinja_env(env)

    header_file = env.get_template('Boundary.tmpl.h').render(**context)
    cpp_file = env.get_template('Boundary.tmpl.cpp').render(**context)
    return header_file, cpp_file


if __name__ == '__main__':
    from lbmpy.boundaries import UBB
    from lbmpy.creationfunctions import create_lb_method

    boundary = UBB(lambda: 0, dim=2)
    method = create_lb_method(stencil='D2Q9', method='srt')
    header, source = create_boundary_class(boundary, method)
    print(source)


