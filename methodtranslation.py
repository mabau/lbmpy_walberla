from waLBerla import lbm

from lbmpy.methods.momentbased import createWithDiscreteMaxwellianEqMoments
from pystencils.cpu.kernelcreation import addOpenMP
import sympy as sp


def createLbmpyMethodFromWalberlaLatticeModel(lm):
    """
    Creates a lbmpy LBM from a waLBerla lattice model
    """
    from lbmpy.methods.momentbased import createSRT, createTRT
    stencil = tuple(lm.directions)

    def getForce():
        dim = len(stencil[0])
        forceAsList = list(lm.forceModel.force())
        return tuple(forceAsList[:dim])

    if type(lm.forceModel) == lbm.forceModels.SimpleConstant:
        forceModel = lbm.forcemodels.Simple(getForce())
    elif type(lm.forceModel) == lbm.forceModels.LuoConstant:
        forceModel = lbm.forcemodels.Luo(getForce())
    elif type(lm.forceModel) == lbm.forceModels.GuoConstant:
        forceModel = lbm.forcemodels.Guo(getForce())
    elif type(lm.forceModel) == lbm.forceModels.NoForce:
        forceModel = None
    else:
        raise NotImplementedError("No such force model in lbmpy")

    commonParams = {'equilibriumAccuracyOrder': lm.equilibriumAccuracyOrder,
                    'compressible': lm.compressible,
                    'forceModel': forceModel}

    if type(lm.collisionModel) == lbm.collisionModels.SRT:
        return createSRT(stencil, lm.collisionModel.omega, **commonParams)
    elif type(lm.collisionModel) == lbm.collisionModels.TRT:
        cm = lm.collisionModel
        return createTRT(stencil, cm.lambda_e, cm.lambda_d, **commonParams)
    elif type(lm.collisionModel) == lbm.collisionModels.D3Q19MRT:
        from lbmpy.moments import MOMENT_SYMBOLS
        x, y, z = MOMENT_SYMBOLS
        s = lm.collisionModel.relaxationRates
        sq = x ** 2 + y ** 2 + z ** 2
        one = sp.Rational(1, 1)
        mrtDef = {
            one: 1, x: 1, y: 1, z: 1,
            3 * x ** 2 - sq: s[9], y ** 2 - z ** 2: s[9], x * y: s[9], y * z: s[9], x * z: s[9],  # [9, 11, 13, 14, 15]
            sq - 1: s[1],  # [1]
            3 * sq ** 2 - 6 * sq + 1: s[2],  # [2]
            (3 * sq - 5) * x: s[4], (3 * sq - 5) * y: s[4], (3 * sq - 5) * z: s[4],  # [4, 6, 8]
            (2 * sq - 3) * (3 * x ** 2 - sq): s[10], (2 * sq - 3) * (y ** 2 - z ** 2): s[10],  # [10, 12]
            (y ** 2 - z ** 2) * x: s[16], (z ** 2 - x ** 2) * y: s[16], (x ** 2 - y ** 2) * z: s[16]  # [16, 17, 18]
        }
        return createWithDiscreteMaxwellianEqMoments(stencil, mrtDef, **commonParams)
    else:
        raise ValueError("Unknown lattice model")


def createWalberlaSweepFromPystencilsKernel(kernel, sourceFieldName='src', destinationFieldName='dst', is2D=False):
    from waLBerla import field
    from pystencils.cpu.cpujit import buildCTypeArgumentList, compileAndLoad
    swapFields = {}

    func = compileAndLoad(kernel)[kernel.functionName]
    func.restype = None

    def f(**kwargs):
        src = kwargs[sourceFieldName]
        sizeInfo = (src.size, src.allocSize, src.layout)
        if sizeInfo not in swapFields:
            swapFields[sizeInfo] = src.cloneUninitialized()
        dst = swapFields[sizeInfo]

        kwargs[sourceFieldName] = field.toArray(src, withGhostLayers=True)
        kwargs[destinationFieldName] = field.toArray(dst, withGhostLayers=True)

        # Since waLBerla does not really support 2D domains a small hack is required here
        if is2D:
            assert kwargs[sourceFieldName].shape[2] in [1, 3]
            assert kwargs[destinationFieldName].shape[2] in [1, 3]
            kwargs[sourceFieldName] = kwargs[sourceFieldName][:, :, 1, :]
            kwargs[destinationFieldName] = kwargs[destinationFieldName][:, :, 1, :]

        args = buildCTypeArgumentList(kernel.parameters, kwargs)
        func(*args)
        src.swapDataPointers(dst)

    return f


def createLbmpySweepFromWalberlaLatticeModel(walberlaLatticeModel, blocks, pdfFieldName, variableSize=False,
                                             doCSE=False, splitInnerLoop=True, openmpThreads=True):
    from lbmpy_old.lbmgenerator import createStreamCollideUpdateRule, createLbmSplitGroups
    from pystencils.cpu import createKernel
    from waLBerla import field

    lbmMethod = createLbmpyMethodFromWalberlaLatticeModel(walberlaLatticeModel)

    numpyField = None
    if not variableSize:
        numpyField = field.toArray(blocks[0][pdfFieldName], withGhostLayers=True)
        dim = len(walberlaLatticeModel.directions[0])
        if dim == 2:
            numpyField = numpyField[:, :, 1, :]

    lbmUpdateRule = createStreamCollideUpdateRule(lbmMethod, numpyField=numpyField, doCSE=doCSE)
    splitGroups = createLbmSplitGroups(lbmMethod, lbmUpdateRule.equations) if splitInnerLoop else []
    funcNode = createKernel(lbmUpdateRule.equations, splitGroups=splitGroups)

    if openmpThreads:
        numThreads = None
        if type(openmpThreads) is int:
            numThreads = openmpThreads
        addOpenMP(funcNode, numThreads=numThreads)

    sweepFunction = createWalberlaSweepFromPystencilsKernel(funcNode, 'src', 'dst', is2D=(lbmMethod.dim == 2))
    sweepFunction = createWalberlaSweepFromPystencilsKernel(funcNode, 'src', 'dst', is2D=(lbmMethod.dim == 2))
    return lambda block: sweepFunction(src=block[pdfFieldName])


def createBoundaryIndexListFromWalberlaFlagField(flagField, stencil, boundaryFlag, fluidFlag):
    import waLBerla as wlb
    from lbmpy.boundaries.createindexlist import createBoundaryIndexList
    flagFieldArr = wlb.field.toArray(flagField, withGhostLayers=True)
    fluidMask = flagField.flag(fluidFlag)
    boundaryMask = flagField.flag(boundaryFlag)
    gl = flagField.nrOfGhostLayers
    dim = len(stencil[0])
    flagFieldArr = flagFieldArr[:, :, :, 0]
    if dim == 2:
        flagFieldArr = flagFieldArr[:, :, gl]

    return createBoundaryIndexList(flagFieldArr, gl, stencil, boundaryMask, fluidMask)

