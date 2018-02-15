from waLBerla import lbm


def createWalberlaLatticeModel(stencil, method, relaxationRates, compressible=False, order=2,
                               forceModel='none', force=(0, 0, 0), **kwargs):

    if method.lower() == 'srt':
        collisionModel = lbm.collisionModels.SRT(relaxationRates[0])
    elif method.lower() == 'trt':
        collisionModel = lbm.collisionModels.TRT(relaxationRates[0], relaxationRates[1])
    elif method.lower() == 'mrt':
        if stencil != 'D3Q19':
            raise ValueError("MRT is available for D3Q19 only in waLBerla")
        collisionModel = lbm.collisionModels.D3Q19MRT(*relaxationRates[1:7])
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
    else:
        raise ValueError("Unknown force model")
    return lbm.makeLatticeModel(stencil, collisionModel, forceModel, compressible, order)

