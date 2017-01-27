from waLBerla import lbm
from pystencils.cpu.kernelcreation import addOpenMP


def createLbmpyMethodFromWalberlaLatticeModel(lm):
    """
    Creates a lbmpy LBM from a waLBerla lattice model
    """
    from lbmpy.methods.momentbased import createSRT, createTRT, createOrthogonalMRT
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
        relaxationRates = [1] + [s[i] for i in (1, 2, 4, 9, 10, 16)]

        nextRelaxationRate = 0

        def relaxationRateGetter(momentGroup):
            nonlocal nextRelaxationRate
            res = relaxationRates[nextRelaxationRate]
            nextRelaxationRate += 1
            return res

        return createOrthogonalMRT(stencil, relaxationRateGetter, **commonParams)
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


def createWalberlaLatticeModel(stencil, method, relaxationRates, compressible=False, order=2,
                               forceModel='none', force=(0, 0, 0)):

    if method.lower() == 'srt':
        collisionModel = lbm.collisionModels.SRT(relaxationRates[0])
    elif method.lower() == 'trt':
        collisionModel = lbm.collisionModels.TRT(relaxationRates[0], relaxationRates[1])
    elif method.lower() == 'mrt':
        if stencil != 'D3Q19':
            raise ValueError("MRT is available for D3Q19 only in waLBerla")
        collisionModel = lbm.collisionModels.D3Q19MRT(*relaxationRates[:6])
    else:
        raise ValueError("Unknown method: " + str(method))

    if len(force) == 2:
        force = (force[0], force[1], 0)

    if forceModel is None or forceModel.lower() == 'none':
        forceModel = lbm.forceModels.NoForce()
    elif forceModel.lower() == 'simple':
        forceModel = lbm.forceModels.SimpleConstant(force)
    elif forceModel.lower() == 'luo':
        forceModel = lbm.forceModels.LuoConstant(force)
    elif forceModel.lower() == 'guo':
        forceModel = lbm.forceModels.GuoConstant(force)

    return lbm.makeLatticeModel(stencil, collisionModel, forceModel, compressible, order)

