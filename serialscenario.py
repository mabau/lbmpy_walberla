from lbmpy_walberla.methodtranslation import *
import waLBerla.lbm as lbm
import waLBerla.field as field
from waLBerla import makeSlice, createUniformBufferedScheme, createUniformBlockGrid


def runForceDrivenChannel2D(force, radius, length, **kwargs):

    kwargs['force'] = tuple([force, 0, 0])

    domainSize = (length, 2 * radius)

    latticeModel = createWalberlaLatticeModel(**kwargs)
    blocks = createUniformBlockGrid(cells=domainSize, periodic=(1, 0, 1))

    # Adding fields
    lbm.addPdfFieldToStorage(blocks, "pdfs", latticeModel, velocityAdaptor="vel", densityAdaptor="rho",
                             initialDensity=1.0)
    field.addFlagFieldToStorage(blocks, 'flags')
    lbm.addBoundaryHandlingToStorage(blocks, 'boundary', 'pdfs', 'flags')

    # Communication
    communication = createUniformBufferedScheme(blocks, latticeModel.communicationStencilName)
    communication.addDataToCommunicate(field.createPackInfo(blocks, 'pdfs'))

    # Setting boundaries
    for block in blocks:
        b = block['boundary']
        if block.atDomainMaxBorder[1]:  # N
            b.forceBoundary('NoSlip', makeSlice[:, -1, :, 'g'])
        if block.atDomainMinBorder[1]:  # S
            b.forceBoundary('NoSlip', makeSlice[:, 0, :, 'g'])

        b.fillWithDomain()

    sweep = lbm.makeCellwiseSweep(blocks, "pdfs", flagFieldID='flags', flagList=['fluid']).streamCollide

    def timeLoop(timeSteps):
        for t in range(timeSteps):
            communication()
            for block in blocks:
                block['boundary']()
            for block in blocks:
                sweep(block)
        fullPdfField = field.toArray(field.gather(blocks, 'pdfs', makeSlice[:, :, :]), withGhostLayers=False)
        density = field.toArray(field.gather(blocks, 'rho', makeSlice[:, :, :]), withGhostLayers=False)
        velocity = field.toArray(field.gather(blocks, 'vel', makeSlice[:, :, :]), withGhostLayers=False)
        fullPdfField = fullPdfField[:, :, 0, :]
        density = density[:, :, 0, 0]
        velocity = velocity[:, :, 0, :2]
        return fullPdfField, density, velocity

    return timeLoop


def runLidDrivenCavity(domainSize, lidVelocity=0.005, **kwargs):
    D = len(domainSize)

    if 'stencil' not in kwargs:
        kwargs['stencil'] = 'D2Q9' if D == 2 else 'D3Q27'

    if D == 2:
        domainSize = (domainSize[0], domainSize[1], 1)

    latticeModel = createWalberlaLatticeModel(**kwargs)
    blocks = createUniformBlockGrid(cells=domainSize, periodic=(1, 1, 1))

    # Adding fields
    lbm.addPdfFieldToStorage(blocks, "pdfs", latticeModel, velocityAdaptor="vel", densityAdaptor="rho",
                             initialDensity=1.0)
    field.addFlagFieldToStorage(blocks, 'flags')
    lbm.addBoundaryHandlingToStorage(blocks, 'boundary', 'pdfs', 'flags')

    # Communication
    communication = createUniformBufferedScheme(blocks, latticeModel.communicationStencilName)
    communication.addDataToCommunicate(field.createPackInfo(blocks, 'pdfs'))

    # Setting boundaries
    for block in blocks:
        b = block['boundary']
        if block.atDomainMaxBorder[1]:  # N
            b.forceBoundary('UBB', makeSlice[:, -1, :, 'g'], {'x': lidVelocity})
        if block.atDomainMinBorder[1]:  # S
            b.forceBoundary('NoSlip', makeSlice[:, 0, :, 'g'])
        if block.atDomainMinBorder[0]:  # W
            b.forceBoundary('NoSlip', makeSlice[0, :, :, 'g'])
        if block.atDomainMaxBorder[0]:  # E
            b.forceBoundary('NoSlip', makeSlice[-1, :, :, 'g'])
        if block.atDomainMinBorder[2] and D == 3:  # T
            b.forceBoundary('NoSlip', makeSlice[:, :, 0, 'g'])
        if block.atDomainMaxBorder[2] and D == 3:  # B
            b.forceBoundary('NoSlip', makeSlice[:, :, -1, 'g'])

        b.fillWithDomain()

    sweep = lbm.makeCellwiseSweep(blocks, "pdfs", flagFieldID='flags', flagList=['fluid']).streamCollide

    def timeLoop(timeSteps):
        for t in range(timeSteps):
            communication()
            for block in blocks:
                block['boundary']()
            for block in blocks:
                sweep(block)
        fullPdfField = field.toArray(field.gather(blocks, 'pdfs', makeSlice[:, :, :]), withGhostLayers=False)
        density = field.toArray(field.gather(blocks, 'rho', makeSlice[:, :, :]), withGhostLayers=False)
        velocity = field.toArray(field.gather(blocks, 'vel', makeSlice[:, :, :]), withGhostLayers=False)
        if D == 2:
            fullPdfField = fullPdfField[:, :, 0, :]
            density = density[:, :, 0, 0]
            velocity = velocity[:, :, 0, :2]
        elif D == 3:
            density = density[:, :, :, 0]

        return fullPdfField, density, velocity

    return timeLoop


if __name__ == '__main__':
    tl = runLidDrivenCavity((20, 20), method='SRT', relaxationRates=[1.4])
    tl(100)
