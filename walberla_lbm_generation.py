import sympy as sp
from sympy.tensor import IndexedBase
from jinja2 import Environment, PackageLoader, Template
import os
import inspect

from pystencils.astnodes import SympyAssignment
from pystencils.equationcollection.equationcollection import EquationCollection
from pystencils.sympyextensions import getSymmetricPart
from pystencils.field import offsetToDirectionString, Field
from pystencils.backends.cbackend import CustomSympyPrinter, CBackend
from pystencils.data_types import TypedSymbol
from pystencils_walberla.sweep import KernelInfo
from pystencils_walberla.jinja_filters import addPystencilsFiltersToJinjaEnv
from pystencils.cpu import addOpenMP, createKernel

from lbmpy.methods.relaxationrates import relaxationRateScaling
from lbmpy.creationfunctions import createLatticeBoltzmannMethod, updateWithDefaultParameters,\
    createLatticeBoltzmannAst, createLatticeBoltzmannUpdateRule
from lbmpy.updatekernels import createStreamPullOnlyKernel    

cppPrinter = CustomSympyPrinter()


REFINEMENT_SCALE_FACTOR = sp.Symbol("levelScaleFactor")


def stencilSwitchStatement(stencil, values):
    templ = Template("""
    using namespace stencil;
    switch( direction ) {
        {% for directionName, value in dirToValueDict.items() -%}
            case {{directionName}}: return {{value}};
        {% endfor -%}
        default:
            WALBERLA_ABORT("Invalid Direction");
    }
    """)

    dirToValueDict = {offsetToDirectionString(d): cppPrinter.doprint(v) for d, v in zip(stencil, values)}
    return templ.render(dirToValueDict=dirToValueDict)


def fieldAndSymbolSubstitute(expr, variablePrefix="lm.", variablesWithoutPrefix=[]):
    variablesWithoutPrefix = [v.name if isinstance(v, sp.Symbol) else v for v in variablesWithoutPrefix]
    substitutions = {}
    for sym in expr.atoms(sp.Symbol):
        if isinstance(sym, Field.Access):
            fa = sym
            prefix = "" if fa.field.name in variablesWithoutPrefix else variablePrefix
            if fa.field.indexDimensions == 0:
                substitutions[fa] = sp.Symbol("%s%s->get(x,y,z)" % (prefix, fa.field.name))
            else:
                assert fa.field.indexDimensions == 1, "waLBerla supports only 0 or 1 index dimensions"
                substitutions[fa] = sp.Symbol("%s%s->get(x,y,z,%s)" % (prefix, fa.field.name, fa.index[0]))
        else:
            if sym.name not in variablesWithoutPrefix:
                substitutions[sym] = sp.Symbol(variablePrefix + sym.name)
    return expr.subs(substitutions)


def expressionToCode(expr, variablePrefix="lm.", variablesWithoutPrefix=[]):
    """
    Takes a sympy expression and creates a C code string from it. Replaces field accesses by
    waLBerla field accesses i.e. field_W^1 -> field->get(-1, 0, 0, 1)
    :param expr: sympy expression
    :param variablePrefix: all variables (and field) are prefixed with this string
                           this is used for member variables mostly
    :param variablesWithoutPrefix: this variables are not prefixed
    :return: code string
    """
    return cppPrinter.doprint(fieldAndSymbolSubstitute(expr, variablePrefix, variablesWithoutPrefix))


def equationsToCode(equations, variablePrefix="lm.", variablesWithoutPrefix=[]):
    def typeEq(eq):
        return eq.subs({s: TypedSymbol(s.name, "double") for s in eq.atoms(sp.Symbol)})

    if isinstance(equations, EquationCollection):
        equations = equations.allEquations

    variablesWithoutPrefix = list(variablesWithoutPrefix)
    cBackend = CBackend()
    result = []
    leftHandSideNames = [e.lhs.name for e in equations]
    for eq in equations:
        assignment = SympyAssignment(typeEq(eq.lhs),
                                     fieldAndSymbolSubstitute(eq.rhs, variablePrefix,
                                                              variablesWithoutPrefix + leftHandSideNames))
        result.append(cBackend(assignment))
    return "\n".join(result)


def generateLatticeModel(latticeModelName=None, optimizationParams={}, refinementScaling=None, method=None, **kwargs):
    """
    Creates a waLBerla lattice model consisting of a source and header file

    :param latticeModelName: name of the generated lattice model class. If None, it is assumed that this function was
                             called from a .gen.py file and the filename is taken as lattice model name
    :param optimizationParams: see documentation of createLatticeBoltzmannAst
    :param kwargs: see documentation of createLatticeBoltzmannAst
    :param refinementScaling: dict from parameter symbol (e.g. relaxationRate, force parameter) to an expression
                              how the parameter scales on refined blocks. The refinement factor is represented by
                              the global symbol REFINEMENT_SCALE_FACTOR
    :param method: optionally pass an already create LB method in here                              
    :return: tuple with code strings (header, sources)
    """
    if latticeModelName is None:
        scriptFileName = inspect.stack()[-1][1]
        if scriptFileName.endswith(".cuda.gen.py"):
            raise ValueError("GPU Lattice Model are not yet supported")
        elif scriptFileName.endswith(".gen.py"):
            fileName = scriptFileName[:-len(".gen.py")]
            latticeModelName = os.path.split(fileName)[1]
        else:
            raise ValueError("Not called from a .gen.py file and latticeModelName is missing")

    if 'fieldLayout' not in optimizationParams:
        # usually a numpy layout is chosen by default i.e. xyzf - which is bad for waLBerla where at least the spatial
        # coordinates should be ordered in reverse direction i.e. zyx
        optimizationParams['fieldLayout'] = 'fzyx'

    params, optParams = updateWithDefaultParameters(kwargs, optimizationParams)

    stencilName = params['stencil']

    if params['forceModel'] != 'none' and params['force'] == (0, 0, 0):
        params['force'] = sp.symbols("force:3")

    params['fieldName'] = 'pdfs'
    params['secondFieldName'] = 'pdfs_tmp'
    if not method:
        method = createLatticeBoltzmannMethod(**params)
    streamCollideUpdate = createLatticeBoltzmannUpdateRule(lbMethod=method, optimizationParams=optParams, **params)
    streamCollideAst = createLatticeBoltzmannAst(updateRule=streamCollideUpdate, optimizationParams=optParams, **params)
    streamCollideAst.functionName = 'kernel_streamCollide'

    params['kernelType'] = 'collideOnly'
    collideOnlyUpdate = createLatticeBoltzmannUpdateRule(lbMethod=method, optimizationParams=optParams, **params)
    collideAst = createLatticeBoltzmannAst(updateRule=collideOnlyUpdate, optimizationParams=optParams, **params)
    collideAst.functionName = 'kernel_collide'

    streamUpdateRule = createStreamPullOnlyKernel(method.stencil, srcFieldName=params['fieldName'], dstFieldName=params['secondFieldName'],
                                                  genericLayout=optParams['fieldLayout'])
    streamAst = createKernel(streamUpdateRule.allEquations)                                                   
    streamAst.functionName = 'kernel_stream'
    
    addOpenMP(streamAst, numThreads=optParams['openMP'])
    addOpenMP(collideAst, numThreads=optParams['openMP'])
    addOpenMP(streamCollideAst, numThreads=optParams['openMP'])

    velSymbols = method.conservedQuantityComputation.firstOrderMomentSymbols
    rhoSym = sp.Symbol("rho")
    pdfsSym = sp.symbols("f_:%d" % (len(method.stencil),))
    velArrSymbols = [IndexedBase(sp.Symbol('u'), shape=(1,))[i] for i in range(len(velSymbols))]
    momentumDensityArrSymbols = [sp.Symbol("md_%d" % (i,)) for i in range(len(velSymbols))]

    equilibrium = method.getEquilibriumTerms().subs({a: b for a, b in zip(velSymbols, velArrSymbols)})
    symmetricEquilibrium = getSymmetricPart(equilibrium, velArrSymbols)
    asymmetricEquilibrium = sp.expand(equilibrium - symmetricEquilibrium)

    forceModel = method.forceModel
    macroscopicVelocityShift = None
    if forceModel:
        if hasattr(forceModel, 'macroscopicVelocityShift'):
            macroscopicVelocityShift = [expressionToCode(e, "lm.", ["rho"])
                                        for e in forceModel.macroscopicVelocityShift(rhoSym)]

    cqc = method.conservedQuantityComputation

    eqInputFromInputEqs = cqc.equilibriumInputEquationsFromInitValues(sp.Symbol("rhoIn"), velArrSymbols)
    densityVelocitySetterMacroscopicValues = equationsToCode(eqInputFromInputEqs, variablesWithoutPrefix=['rhoIn', 'u'])
    momentumDensityGetter = cqc.outputEquationsFromPdfs(pdfsSym, {'density': rhoSym,
                                                                  'momentumDensity': momentumDensityArrSymbols})

    context = {
        'className': latticeModelName,
        'stencilName': stencilName,
        'D': method.dim,
        'Q': len(method.stencil),
        'compressible': 'true' if params['compressible'] else 'false',
        'weights': ",".join(str(w.evalf()) for w in method.weights),
        'inverseWeights': ",".join(str((1/w).evalf()) for w in method.weights),

        'equilibriumAccuracyOrder': params['equilibriumAccuracyOrder'],

        'equilibriumFromDirection': stencilSwitchStatement(method.stencil, equilibrium),
        'symmetricEquilibriumFromDirection': stencilSwitchStatement(method.stencil, symmetricEquilibrium),
        'asymmetricEquilibriumFromDirection': stencilSwitchStatement(method.stencil, asymmetricEquilibrium),
        'equilibrium': [cppPrinter.doprint(e) for e in equilibrium],

        'macroscopicVelocityShift': macroscopicVelocityShift,
        'densityGetters': equationsToCode(cqc.outputEquationsFromPdfs(pdfsSym, {"density": rhoSym}),
                                          variablesWithoutPrefix=[e.name for e in pdfsSym]),
        'momentumDensityGetter': equationsToCode(momentumDensityGetter, variablesWithoutPrefix=pdfsSym),
        'densityVelocitySetterMacroscopicValues': densityVelocitySetterMacroscopicValues,

        'refinementLevelScaling': refinementScaling,

        'streamCollideKernel': KernelInfo(streamCollideAst, ['pdfs_tmp'], [('pdfs', 'pdfs_tmp')]),
        'collideKernel': KernelInfo(collideAst, [], []),
        'streamKernel': KernelInfo(streamAst, ['pdfs_tmp'], [('pdfs', 'pdfs_tmp')]),
        'target': 'cpu',
        'namespace': 'lbm',
    }

    env = Environment(loader=PackageLoader('lbmpy_walberla'))
    addPystencilsFiltersToJinjaEnv(env)

    headerFile = env.get_template('LatticeModel.tmpl.h').render(**context)
    cppFile = env.get_template('LatticeModel.tmpl.cpp').render(**context)
    return headerFile, cppFile


def generateLatticeModelFiles(**kwargs):
    """
    :param kwargs: see documentation of createLatticeBoltzmannAst, additionally 
                   an instance of RefinementScaling can be passed with the 'refinementScaling' keyword
    """
    scriptFileName = inspect.stack()[-1][1]
    if scriptFileName.endswith(".cuda.gen.py"):
        raise ValueError("GPU Lattice Model are not yet supported")
    elif scriptFileName.endswith(".gen.py"):
        fileName = scriptFileName[:-len(".gen.py")]
        latticeModelName = os.path.split(fileName)[1]
    else:
        raise ValueError("Not called from a .gen.py file and latticeModelName is missing")

    header, sources = generateLatticeModel(latticeModelName, **kwargs)

    with open(latticeModelName + ".h", 'w') as f:
        f.write(header)
    with open(latticeModelName + ".cpp", 'w') as f:
        f.write(sources)


class RefinementScaling:
    levelScaleFactor = sp.Symbol("levelScaleFactor")

    def __init__(self):
        self.scalings = []

    def addStandardRelaxationRateScaling(self, viscosityRelaxationRate):
        self.addScaling(viscosityRelaxationRate, relaxationRateScaling)

    def addForceScaling(self, forceParameter):
        self.addScaling(forceParameter, lambda param, factor:  param * factor)

    def addScaling(self, parameter, scalingRule):
        """
        Adds a scaling rule, how parameters on refined blocks are modified

        :param parameter: parameter to modify: may either be a Field, Field.Access or a Symbol
        :param scalingRule: function taking the parameter to be scaled as symbol and the scaling factor i.e.
                            how much finer the current block is compared to coarsest resolution
        """
        if isinstance(parameter, Field):
            field = parameter
            name = field.name
            if field.indexDimensions > 0:
                scalingType = 'fieldWithF'
                fieldAccess = field(sp.Symbol("f"))
            else:
                scalingType = 'fieldXYZ'
                fieldAccess = field.center()
            expr = scalingRule(fieldAccess, self.levelScaleFactor)
            self.scalings.append((scalingType, name, expressionToCode(expr, '')))
        elif isinstance(parameter, Field.Access):
            fieldAccess = parameter
            expr = scalingRule(fieldAccess, self.levelScaleFactor)
            name = fieldAccess.field.name
            self.scalings.append(('fieldXYZ', name, expressionToCode(expr, '')))
        elif isinstance(parameter, sp.Symbol):
            expr = scalingRule(parameter, self.levelScaleFactor)
            self.scalings.append(('normal', parameter.name, expressionToCode(expr, '')))
        elif isinstance(parameter, list) or isinstance(parameter, tuple):
            for p in parameter:
                self.addScaling(p, scalingRule)
        else:
            raise ValueError("Invalid value for viscosityRelaxationRate")


if __name__ == '__main__':
    from pystencils import Field
    omega = sp.Symbol("omega")

    forceField = Field.createGeneric('force', spatialDimensions=3, indexDimensions=1, layout='c')
    force = [forceField(0), forceField(1), forceField(2)]

    scaling = RefinementScaling()
    scaling.addStandardRelaxationRateScaling(omega)
    scaling.addForceScaling(forceField)

    header, cpp = generateLatticeModel(latticeModelName='GenLM', method='srt', stencil='D3Q19',
                                       forceModel='guo', force=force, relaxationRates=[omega],
                                       refinementScaling=scaling)

    with open("/home/martin/dev/walberla/apps/tutorials/lbm/GenLM.h", 'w') as f:
        f.write(header)
    with open("/home/martin/dev/walberla/apps/tutorials/lbm/GenLM.cpp", 'w') as f:
        f.write(cpp)
