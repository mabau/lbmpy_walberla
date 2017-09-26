//======================================================================================================================
//
//  This file is part of waLBerla. waLBerla is free software: you can
//  redistribute it and/or modify it under the terms of the GNU General Public
//  License as published by the Free Software Foundation, either version 3 of
//  the License, or (at your option) any later version.
//
//  waLBerla is distributed in the hope that it will be useful, but WITHOUT
//  ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
//  FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
//  for more details.
//
//  You should have received a copy of the GNU General Public License along
//  with waLBerla (see COPYING.txt). If not, see <http://www.gnu.org/licenses/>.
//
//! \\author Martin Bauer <martin.bauer@fau.de>
//======================================================================================================================

#include <cmath>

#include "core/DataTypes.h"
#include "core/Macros.h"
#include "lbm/field/PdfField.h"
#include "{{className}}.h"

#ifdef _MSC_VER
#  pragma warning( disable : 4458 )
#endif

{% if target is equalto 'cpu' -%}
#define FUNC_PREFIX
{%- elif target is equalto 'gpu' -%}
#define FUNC_PREFIX __global__
{%- endif %}

using namespace std;

namespace walberla {
namespace {{namespace}} {

{{streamCollideKernel|generateDefinition}}
{{collideKernel|generateDefinition}}


const real_t {{className}}::w[{{Q}}] = { {{weights}} };
const real_t {{className}}::wInv[{{Q}}] = { {{inverseWeights}} };

void {{className}}::Sweep::streamCollide( IBlock * block, const uint_t numberOfGhostLayersToInclude )
{
    {{streamCollideKernel|generateBlockDataToFieldExtraction(parameters=['pdfs', 'pdfs_tmp'])|indent(4)}}

    auto & lm = dynamic_cast< lbm::PdfField<{{className}}> * > (pdfs)->latticeModel();
    lm.configureBlock(block);

    {{streamCollideKernel|generateRefsForKernelParameters(prefix='lm.', parametersToIgnore=['pdfs', 'pdfs_tmp'])|indent(4) }}
    {{streamCollideKernel|generateCall('cell_idx_c(numberOfGhostLayersToInclude)')|indent(4)}}
    {{streamCollideKernel|generateSwaps|indent(4)}}
}

void {{className}}::Sweep::collide( IBlock * block, const uint_t numberOfGhostLayersToInclude )
{
    WALBERLA_ASSERT(numberOfGhostLayersToInclude == 0, "Not implemented yet");

    {{collideKernel|generateBlockDataToFieldExtraction(parameters=['pdfs'])|indent(4)}}

    auto & lm = dynamic_cast< lbm::PdfField<{{className}}> * > (pdfs)->latticeModel();
    lm.configureBlock(block);

    {{collideKernel|generateRefsForKernelParameters(prefix='lm.', parametersToIgnore=['pdfs', 'pdfs_tmp'])|indent(4) }}
    {{collideKernel|generateCall('cell_idx_c(numberOfGhostLayersToInclude)')|indent(4)}}
    {{collideKernel|generateSwaps|indent(4)}}
}


} // namespace {{namespace}}
} // namespace walberla




// Buffer Packing

namespace walberla {
namespace mpi {

mpi::SendBuffer & operator<< (mpi::SendBuffer & buf, const ::walberla::{{namespace}}::{{className}} & lm)
{
    buf << lm.currentLevel;
    return buf;
}

mpi::RecvBuffer & operator>> (mpi::RecvBuffer & buf, ::walberla::{{namespace}}::{{className}} & lm)
{
    buf >> lm.currentLevel;
    return buf;
}


} // namespace mpi
} // namespace walberla

