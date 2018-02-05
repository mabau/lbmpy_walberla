import numpy as np
from jinja2 import Environment, PackageLoader

from pystencils import Field, FieldType
from pystencils.data_types import createType, TypedSymbol
from pystencils_walberla.jinja_filters import addPystencilsFiltersToJinjaEnv

from lbmpy.boundaries.boundary_kernel import generateIndexBoundaryKernelGeneric
from lbmpy.boundaries.createindexlist import numpyDataTypeForBoundaryObject, \
    boundaryIndexArrayCoordinateNames, directionMemberName
from lbmpy_walberla.walberla_lbm_generation import KernelInfo


def structFromNumpyDataType(structName, numpyDtype):
    result = "struct %s { \n" % (structName,)

    equalityCompare = []
    constructorParams = []
    constructorInitializerList = []
    for name, (subType, offset) in numpyDtype.fields.items():
        pystencilsType = createType(subType)
        result += "    %s %s;\n" % (pystencilsType, name)
        if name in boundaryIndexArrayCoordinateNames or name == directionMemberName:
            constructorParams.append("%s %s_" % (pystencilsType, name))
            constructorInitializerList.append("%s(%s_)" % (name, name))
        else:
            constructorInitializerList.append("%s()" % name)
        if pystencilsType.is_float():
            equalityCompare.append("floatIsEqual(%s, o.%s)" % (name, name))
        else:
            equalityCompare.append("%s == o.%s" % (name, name))

    result += "    %s(%s) : %s {}\n" % (structName, ", ".join(constructorParams), ", ".join(constructorInitializerList))
    result += "    bool operator==(const %s & o) const {\n        return %s;\n    }\n" % \
              (structName, " && ".join(equalityCompare))
    result += "};\n"
    return result


def createBoundaryClass(boundaryObject, lbMethod, doublePrecision=True, target='cpu'):
    structName = "IndexInfo"
    indexStructDtype = numpyDataTypeForBoundaryObject(boundaryObject, lbMethod.dim)

    pdfField = Field.createGeneric('pdfs', lbMethod.dim, np.float64 if doublePrecision else np.float32,
                                   indexDimensions=1, layout='fzyx', indexShape=[len(lbMethod.stencil)])

    indexField = Field('indexVector', FieldType.INDEXED, indexStructDtype, layout=[0],
                       shape=(TypedSymbol("indexVectorSize", createType(np.int64)), 1), strides=(1, 1))

    kernel = generateIndexBoundaryKernelGeneric(pdfField, indexField, lbMethod, boundaryObject, target=target)

    stencilInfo = [(i, ", ".join([str(e) for e in d])) for i, d in enumerate(lbMethod.stencil)]

    context = {
        'className': boundaryObject.name,
        'StructName': structName,
        'StructDeclaration': structFromNumpyDataType(structName, indexStructDtype),
        'kernel': KernelInfo(kernel, [], []),
        'stencilInfo': stencilInfo,
        'dim': lbMethod.dim,
        'target': target,
        'namespace': 'lbm',
    }

    env = Environment(loader=PackageLoader('lbmpy_walberla'))
    addPystencilsFiltersToJinjaEnv(env)

    headerFile = env.get_template('Boundary.tmpl.h').render(**context)
    cppFile = env.get_template('Boundary.tmpl.cpp').render(**context)
    return headerFile, cppFile


if __name__ == '__main__':
    from lbmpy.boundaries import NoSlip, UBB
    from lbmpy.creationfunctions import createLatticeBoltzmannMethod

    boundary = UBB(lambda: 0, dim=2)
    method = createLatticeBoltzmannMethod(stencil='D2Q9', method='srt')
    header, source = createBoundaryClass(boundary, method)
    print(source)


