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
#include "lbm/sweeps/Streaming.h"
#include "{{class_name}}.h"

{% for header in headers %}
#include {{header}}
{% endfor %}


#ifdef _MSC_VER
#  pragma warning( disable : 4458 )
#endif

{% if target is equalto 'cpu' -%}
#define FUNC_PREFIX
{%- elif target is equalto 'gpu' -%}
#define FUNC_PREFIX __global__
{%- endif %}

#ifdef WALBERLA_CXX_COMPILER_IS_GNU
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-variable"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wshadow"
#endif

#ifdef WALBERLA_CXX_COMPILER_IS_CLANG
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-parameter"
#pragma clang diagnostic ignored "-Wshadow"
#endif


using namespace std;

namespace walberla {
namespace {{namespace}} {

{{stream_collide_kernel|generate_definition}}
{{collide_kernel|generate_definition}}
{{stream_kernel|generate_definition}}


const real_t {{class_name}}::w[{{Q}}] = { {{weights}} };
const real_t {{class_name}}::wInv[{{Q}}] = { {{inverse_weights}} };

void {{class_name}}::Sweep::streamCollide( IBlock * block, const uint_t numberOfGhostLayersToInclude )
{
    {{stream_collide_kernel|generate_block_data_to_field_extraction(parameters=['pdfs', 'pdfs_tmp'])|indent(4)}}

    auto & lm = dynamic_cast< lbm::PdfField<{{class_name}}> * > (pdfs)->latticeModel();
    lm.configureBlock(block);

    {{stream_collide_kernel|generate_refs_for_kernel_parameters(prefix='lm.', parameters_to_ignore=['pdfs', 'pdfs_tmp'])|indent(4) }}
    {{stream_collide_kernel|generate_call('cell_idx_c(numberOfGhostLayersToInclude)')|indent(4)}}
    {{stream_collide_kernel|generate_swaps|indent(4)}}
}

void {{class_name}}::Sweep::collide( IBlock * block, const uint_t numberOfGhostLayersToInclude )
{
   {{collide_kernel|generate_block_data_to_field_extraction(parameters=['pdfs'])|indent(4)}}

    auto & lm = dynamic_cast< lbm::PdfField<{{class_name}}> * > (pdfs)->latticeModel();
    lm.configureBlock(block);

    {{collide_kernel|generate_refs_for_kernel_parameters(prefix='lm.', parameters_to_ignore=['pdfs', 'pdfs_tmp'])|indent(4) }}
    {{collide_kernel|generate_call('cell_idx_c(numberOfGhostLayersToInclude)')|indent(4)}}
}


void {{class_name}}::Sweep::stream( IBlock * block, const uint_t numberOfGhostLayersToInclude )
{
    {{stream_kernel|generate_block_data_to_field_extraction(parameters=['pdfs', 'pdfs_tmp'])|indent(4)}}

    {{stream_kernel|generate_call('cell_idx_c(numberOfGhostLayersToInclude)')|indent(4)}}
    
    {{stream_kernel|generate_swaps|indent(4)}}
}


} // namespace {{namespace}}
} // namespace walberla




// Buffer Packing

namespace walberla {
namespace mpi {

mpi::SendBuffer & operator<< (mpi::SendBuffer & buf, const ::walberla::{{namespace}}::{{class_name}} & lm)
{
    buf << lm.currentLevel;
    return buf;
}

mpi::RecvBuffer & operator>> (mpi::RecvBuffer & buf, ::walberla::{{namespace}}::{{class_name}} & lm)
{
    buf >> lm.currentLevel;
    return buf;
}


} // namespace mpi
} // namespace walberla

#ifdef WALBERLA_CXX_COMPILER_IS_GNU
#pragma GCC diagnostic pop
#endif

#ifdef WALBERLA_CXX_COMPILER_IS_CLANG
#pragma clang diagnostic pop
#endif
